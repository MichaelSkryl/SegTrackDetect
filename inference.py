import argparse
import os
import json
import time
import logging

from glob import glob
from tqdm import tqdm
from statistics import mean
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import WindowDetectionDataset, ROIDataset, DirectoryDataset
from drawing import make_vis

from rois import ROIModule
from detector import Detector, overlapping_box_suppression
from detector.aggregation import xyxy2xywh

# ---------------------------------------------------------------------------
# Bystander-analysis helpers
# ---------------------------------------------------------------------------
# Relative-area buckets from Kos et al. 2022 (Table 1 of the paper). Used to
# decide which size class a surviving detection contributes to.
_SIZE_BUCKETS = [
    ('micro',  0.0000, 0.0038),
    ('v-tiny', 0.0038, 0.0152),
    ('tiny',   0.0152, 0.0305),
    ('small',  0.0305, 0.0610),
    ('medium', 0.0610, 0.1829),
    ('large',  0.1829, 1.0001),
]


def _size_bucket_for(rel_area):
    """Return the size-bucket name for a detection of relative area `rel_area`."""
    for name, lo, hi in _SIZE_BUCKETS:
        if lo <= rel_area < hi:
            return name
    return 'large'


def _new_bystander_stats():
    """Fresh accumulator for bystander analysis."""
    return {
        'overall':      {'standard': 0, 'oc_self': 0, 'oc_bystander': 0},
        'by_size':      defaultdict(lambda: {'standard': 0, 'oc_self': 0, 'oc_bystander': 0}),
        'per_sequence': defaultdict(lambda: {'standard': 0, 'oc_self': 0, 'oc_bystander': 0}),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # models
    parser.add_argument('--roi_model', type=str, default="SDS_large", help='ROI Estimation model name. Must be defined in rois.estimator.configs.ESTIMATOR_MODELS')
    parser.add_argument('--det_model', type=str, default="SDS", help='Detection model name. Must be defined in detector.configs.DETECTION_MODELS')
    parser.add_argument('--tracker', type=str, default="sort", help='Tracker name. Must be defined in rois.predictor.configs.PREDICTOR_MODELS')

    # dataset
    parser.add_argument('--data_root', type=str, help='Data root for custom dataset.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use.')
    parser.add_argument('--flist', type=str, help='If provided, infer images listed in flist.txt; if not, infer split images.')
    parser.add_argument('--name', type=str, help='Name for img list provided in flist.txt')

    # ROI
    parser.add_argument('--bbox_type', type=str, default='sorted', choices=['all', 'naive', 'sorted'], help='Type of detection bounding boxes filtering method.')
    parser.add_argument('--allow_resize', default=False, action='store_true', help='Allow resizing of detection sub-windows.')
    
    # Temporal
    parser.add_argument('--use_temporal', type=str, default='none',
                        choices=['none', 'post_unet', 'bottleneck', 'dual', 'postdecoder', 'multiscale'],
                        help='Temporal mode for ROI estimation.')
    parser.add_argument('--temporal_hidden', type=int, default=16, help='Hidden channels for ConvGRU.')
    parser.add_argument('--temporal_ks', type=int, default=3, help='Kernel size for ConvGRU.')
    parser.add_argument('--temporal_weights', type=str, default=None, help='Path to trained temporal weights.')
    parser.add_argument('--insertion_point', type=int, default=5, choices=[2, 3, 4, 5],
                        help='Encoder layer for ConvGRU insertion (bottleneck mode).')

    # Dual-GRU (Idea 1)
    parser.add_argument('--shallow_hidden', type=int, default=16,
                        help='Hidden channels for shallow GRU (dual mode).')
    parser.add_argument('--shallow_point', type=int, default=2, choices=[2, 3, 4],
                        help='Encoder layer for shallow GRU (dual mode).')

    # Temporal Detection Heatmap (Idea 3)
    parser.add_argument('--use_heatmap', default=False, action='store_true',
                        help='Enable temporal detection heatmap for tiny objects.')
    parser.add_argument('--heatmap_decay', type=float, default=0.85,
                        help='Heatmap decay per frame.')
    parser.add_argument('--heatmap_tiny_threshold', type=float, default=0.01,
                        help='Relative size threshold for heatmap boosting.')
    parser.add_argument('--heatmap_activation_threshold', type=float, default=0.3,
                        help='Minimum heatmap value to activate.')
                
    # Adaptive windowing        
    parser.add_argument('--use_adaptive_windowing', default=False, action='store_true',
                        help='Enable object-centric windows for tiny tracked objects.')
    parser.add_argument('--aw_tiny_threshold', type=float, default=0.01,
                        help='Relative size threshold for adaptive windowing.')
    parser.add_argument('--aw_min_window_ratio', type=float, default=0.5,
                        help='Minimum window size as fraction of detector input.')
    parser.add_argument('--aw_padding_factor', type=float, default=2.0,
                        help='Padding multiplier around tracked object.')
    parser.add_argument('--aw_max_extra_windows', type=int, default=5,
                        help='Maximum extra object-centric windows per frame.')

    # general
    parser.add_argument('--cpu', default=False, action='store_true', help='Use CPU for inference.')
    parser.add_argument('--out_dir', type=str, default='detections', help='Output directory for results.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode for visualization.')
    parser.add_argument('--vis_conf_th', type=float, default=0.3, help='Confidence threshold for visualization.')
        
    # OBS
    parser.add_argument('--obs_iou_th', type=float, default=0.7, help='IoU threshold for Overlapping Box Suppression.')
    args = parser.parse_args()
    

   # Create output directory and save arguments to JSON file
    os.makedirs(args.out_dir, exist_ok=False)
    with open(os.path.join(args.out_dir, "args.json"), 'w', encoding='utf-8') as f:
        info = {**vars(args)}
        json.dump(info, f, ensure_ascii=False, indent=4)

    if args.debug:
        debug_dir = f'{args.out_dir}/vis'
        os.makedirs(debug_dir, exist_ok=True)

    # --- Adaptive-windowing diagnostics → file (no console spam) ----------
    # The 'adaptive_windowing' logger is used by rois/fusion.py and
    # rois/windows_proposals.py. By default it has only a NullHandler, so
    # nothing leaks unless we attach a FileHandler here.
    aw_logger = logging.getLogger('adaptive_windowing')
    if args.use_adaptive_windowing:
        aw_logger.setLevel(logging.INFO)
        aw_log_path = os.path.join(args.out_dir, 'aw_debug.log')
        aw_handler = logging.FileHandler(aw_log_path, mode='w', encoding='utf-8')
        aw_handler.setFormatter(logging.Formatter('%(message)s'))
        aw_logger.addHandler(aw_handler)
        aw_logger.propagate = False    # don't bubble up to root → console
        aw_logger.info(f"# Adaptive-windowing debug log")
        aw_logger.info(f"# args: tiny_threshold={args.aw_tiny_threshold} "
                       f"min_window_ratio={args.aw_min_window_ratio} "
                       f"padding_factor={args.aw_padding_factor} "
                       f"max_extra_windows={args.aw_max_extra_windows}")
        print(f"[AW] diagnostics → {aw_log_path}")

    # Bystander analysis: per-detection classification (standard / oc_self / oc_bystander).
    # Only populated when --use_adaptive_windowing is on.
    bystander_stats = _new_bystander_stats() if args.use_adaptive_windowing else None
    # ----------------------------------------------------------------------
    

    # Get dataset
    ds = DirectoryDataset(
        data_root = args.data_root,
        split = args.split,
        flist = args.flist,
        name = args.name,
    )
    seq2images = ds.seq2images
    print(args.allow_resize)
    # Get models
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 and not args.cpu else 'cpu'
    detector = Detector(args.det_model, device)
    roi_extractor = ROIModule(
        tracker_name = args.tracker,
        estimator_name = args.roi_model,
        is_sequence = ds.is_sequential,
        device = device,
        bbox_type = args.bbox_type,
        allow_resize = args.allow_resize,
        # Temporal (bottleneck / post_unet)
        use_temporal = args.use_temporal,
        temporal_hidden = args.temporal_hidden,
        temporal_ks = args.temporal_ks,
        temporal_weights = args.temporal_weights,
        insertion_point = args.insertion_point,
        # Dual-GRU (Idea 1)
        shallow_hidden = args.shallow_hidden,
        shallow_point = args.shallow_point,
        # Heatmap (Idea 3)
        use_heatmap = args.use_heatmap,
        heatmap_decay = args.heatmap_decay,
        heatmap_tiny_threshold = args.heatmap_tiny_threshold,
        heatmap_activation_threshold = args.heatmap_activation_threshold,
        # Adaptive Windowing
        use_adaptive_windowing = args.use_adaptive_windowing,
        aw_tiny_threshold = args.aw_tiny_threshold,
        aw_min_window_ratio = args.aw_min_window_ratio,
        aw_padding_factor = args.aw_padding_factor,
        aw_max_extra_windows = args.aw_max_extra_windows,
    )

    # Save configurations
    detector_config = detector.get_config_dict()
    roi_extractor_config = roi_extractor.get_config_dict()
    with open(os.path.join(args.out_dir, "configs.json"), 'w', encoding='utf-8') as f:
        config = {**roi_extractor_config, **detector_config}
        json.dump(config, f, ensure_ascii=False, indent=4)

    
    # Inference
    annotations = []
    all_images = 0

    times = defaultdict(list)
    window_log = []
    
    # Detector-call counting (to support FPS explanation in the paper)
    det_call_stats = {
        'frames_with_detections': 0,
        'frames_skipped': 0,
        'windows_per_frame': [],
        'inference_calls_total': 0,}

    for seq_name, seq_flist in tqdm(seq2images.items()):
        seq_flist = sorted(seq_flist)

        roi_extractor.reset_predictor() # new tracker for each sequence
        if args.use_adaptive_windowing:
            aw_logger.info(f"\n========== sequence: {seq_name} "
                           f"({len(seq_flist)} frames) ==========")
        
        dataset = ROIDataset(seq_flist, ds, roi_extractor.estimator.input_size, roi_extractor.estimator.preprocess(**roi_extractor.estimator.preprocess_args))
        all_images += len(dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

        with torch.inference_mode():
            for i, (img, metadata) in tqdm(enumerate(dataloader)):

                start_batch = time.time()
                img = img.to(device)
                img_roi = dataset.roi_transform(img)
                original_shape = metadata['coco']['height'].item(), metadata['coco']['width'].item()

                # get detection windows from ROI
                det_bboxes = roi_extractor.get_fused_roi(
                    frame_id = i,
                    img_tensor = img_roi,
                    orig_shape = original_shape,
                    det_shape = detector.input_size,  
                )
                times['roi'].append(time.time()-start_batch)
                window_log.append({
                    'image_id': int(metadata['coco']['id'].item()),
                    'windows': np.asarray(det_bboxes).tolist(),})

                if len(det_bboxes) > 0:

                    t1 = time.time()
                    det_dataset =  WindowDetectionDataset(img, ds, det_bboxes, detector.input_size)
                    img_det, det_metadata = det_dataset.get_batch()
                    times['det_get_batch'].append(time.time()-t1)


                    t1 = time.time()
                    n_windows = img_det.shape[0]
                    det_call_stats['frames_with_detections'] += 1
                    det_call_stats['windows_per_frame'].append(n_windows)
                    det_call_stats['inference_calls_total'] += n_windows
                    detections = detector.get_detections(img_det)
                    times['det_infer'].append(time.time()-t1)

                    t1 = time.time()
                    img_det, img_win = detector.postprocess_detections(detections, det_metadata)
                    times['det_postproc'].append(time.time()-t1)

                    # --- Bystander pre-OBS classification ---
                    # Tag each detection by which type of window produced it.
                    # We must do this BEFORE OBS because OBS does not preserve
                    # the per-detection window mapping otherwise.
                    pre_obs_cls = None
                    if bystander_stats is not None and len(img_det) > 0:
                        oc_w = getattr(roi_extractor, 'last_oc_windows', None)
                        oc_s = getattr(roi_extractor, 'last_oc_sources', None)
                        img_win_np = img_win.detach().cpu().numpy()
                        img_det_np = img_det.detach().cpu().numpy()
                        pre_obs_cls = []
                        for di in range(len(img_det_np)):
                            det_cx = (img_det_np[di, 0] + img_det_np[di, 2]) / 2.0
                            det_cy = (img_det_np[di, 1] + img_det_np[di, 3]) / 2.0
                            win = img_win_np[di]
                            cls = 'standard'
                            if oc_w is not None and len(oc_w) > 0:
                                for oci in range(len(oc_w)):
                                    if (int(win[0]) == int(oc_w[oci, 0]) and
                                        int(win[1]) == int(oc_w[oci, 1]) and
                                        int(win[2]) == int(oc_w[oci, 2]) and
                                        int(win[3]) == int(oc_w[oci, 3])):
                                        src = oc_s[oci]
                                        if (src[0] <= det_cx <= src[2] and
                                            src[1] <= det_cy <= src[3]):
                                            cls = 'oc_self'
                                        else:
                                            cls = 'oc_bystander'
                                        break
                            pre_obs_cls.append(cls)

                    t1 = time.time()
                    # Overlapping Box Suppression
                    if bystander_stats is not None:
                        img_det, del_mask = overlapping_box_suppression(
                            img_win, img_det, th=args.obs_iou_th, return_mask=True,
                        )
                    else:
                        img_det = overlapping_box_suppression(img_win, img_det, th=args.obs_iou_th)
                    times['obs'].append(time.time()-t1)

                    # --- Bystander post-OBS tally ---
                    if bystander_stats is not None and pre_obs_cls is not None:
                        del_np = del_mask.detach().cpu().numpy().astype(bool)
                        surviving_cls = [c for i, c in enumerate(pre_obs_cls) if not del_np[i]]
                        if len(img_det) > 0:
                            img_det_np2 = img_det.detach().cpu().numpy()
                            img_h, img_w = original_shape
                            img_area = float(img_h) * float(img_w)
                            for di in range(len(img_det_np2)):
                                cls = surviving_cls[di] if di < len(surviving_cls) else 'standard'
                                det_w = max(0.0, img_det_np2[di, 2] - img_det_np2[di, 0])
                                det_h = max(0.0, img_det_np2[di, 3] - img_det_np2[di, 1])
                                rel = (det_w * det_h) / img_area if img_area > 0 else 0.0
                                bucket = _size_bucket_for(rel)
                                bystander_stats['overall'][cls] += 1
                                bystander_stats['by_size'][bucket][cls] += 1
                                bystander_stats['per_sequence'][seq_name][cls] += 1

                else:
                    det_call_stats['frames_skipped'] += 1
                    det_call_stats['windows_per_frame'].append(0)
                    img_det = torch.empty((0,6))

                t1 = time.time()

                # Update temporal heatmap with current detections (Idea 3)
                if args.use_heatmap:
                    roi_extractor.update_heatmap(
                        detections=img_det.detach().cpu().numpy(),
                        orig_shape=original_shape,
                    )

                roi_extractor.update_predictor(img_det.detach().cpu().numpy()[:, :-1])
                                                    
                if args.debug:
                    frame = cv2.imread(metadata['image_path'][0])
                    estim_mask, pred_mask = roi_extractor.get_masks(frame.shape[:2])

                    # Separate OC windows from standard so they don't all look
                    # identical in the rendered debug image.
                    oc_w = getattr(roi_extractor, 'last_oc_windows', None)
                    oc_s = getattr(roi_extractor, 'last_oc_sources', None)
                    has_oc = oc_w is not None and len(oc_w) > 0

                    # Standard windows only = full detection set minus the OC tail.
                    # _merge_adaptive_windows concatenates OC windows at the end
                    # of `detection_windows`, so trimming the last len(oc_w) rows
                    # gives the pure standard windows for make_vis.
                    if has_oc and len(det_bboxes) >= len(oc_w):
                        std_windows = det_bboxes[:len(det_bboxes) - len(oc_w)]
                    else:
                        std_windows = det_bboxes

                    frame = make_vis(frame, estim_mask, pred_mask, std_windows,
                                     img_det, detector.config['classes'],
                                     detector.config['colors'], args.vis_conf_th)

                    # Adaptive-window overlay: yellow window + magenta source bbox
                    # + line linking the two centers, with an "OC" tag.
                    if has_oc:
                        for w, s in zip(oc_w, oc_s):
                            cv2.rectangle(frame, (int(w[0]), int(w[1])),
                                          (int(w[2]), int(w[3])), (0, 255, 255), 3)
                            cv2.rectangle(frame, (int(s[0]), int(s[1])),
                                          (int(s[2]), int(s[3])), (255, 0, 255), 2)
                            wc = (int((w[0] + w[2]) / 2), int((w[1] + w[3]) / 2))
                            sc = (int((s[0] + s[2]) / 2), int((s[1] + s[3]) / 2))
                            cv2.line(frame, wc, sc, (0, 255, 255), 1)
                            cv2.circle(frame, sc, 4, (255, 0, 255), -1)
                            cv2.putText(frame, "OC", (int(w[0]) + 4, int(w[1]) + 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 255), 2)

                    out_fname = f"{debug_dir}/{seq_name}/{os.path.basename(metadata['image_path'][0])}"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    cv2.imwrite(out_fname, frame)
                
                img_det[:,:4] = xyxy2xywh(img_det[:,:4])
                for p in img_det.tolist():
                    annotations.append(
                        {
                            "id": len(annotations), 
                            "image_id": int(metadata['coco']['id'].item()),
                            "category_id": int(p[-1]),
                            "bbox": [round(x, 3) for x in p[:4]],
                            "area": p[2] * p[3],
                            "score": round(p[4], 5),
                            "iscrowd": 0,
                        }
                    )

                end_batch = time.time()
                times['save_dets'].append(time.time()-t1)
                times['total'].append(end_batch-start_batch)

    times = {k: sum(v)/all_images for k,v in times.items()}
    times['fps'] = 1/times['total']
    times = pd.DataFrame(times, index=[0])
    times.to_csv(os.path.join(args.out_dir, 'times.csv'), index=False)
    with open(os.path.join(args.out_dir, 'windows_per_frame.json'), 'w') as f:
        json.dump(window_log, f)

        
    with open(os.path.join(args.out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
    
    # Save detector-call statistics
    import statistics
    wpf = det_call_stats['windows_per_frame']
    det_call_stats['mean_windows_per_frame'] = sum(wpf) / max(1, len(wpf))
    det_call_stats['median_windows_per_frame'] = statistics.median(wpf) if wpf else 0
    det_call_stats['min_windows_per_frame'] = min(wpf) if wpf else 0
    det_call_stats['max_windows_per_frame'] = max(wpf) if wpf else 0
    # Drop the raw list to keep the JSON manageable
    del det_call_stats['windows_per_frame']
    with open(os.path.join(args.out_dir, 'detector_call_stats.json'), 'w') as f:
        json.dump(det_call_stats, f, indent=2)
    print(f"\nDetector inference summary:")
    print(f"  Total inference calls: {det_call_stats['inference_calls_total']}")
    print(f"  Mean windows per frame: {det_call_stats['mean_windows_per_frame']:.2f}")
    print(f"  Median: {det_call_stats['median_windows_per_frame']:.1f}, "f"min/max: {det_call_stats['min_windows_per_frame']}/" f"{det_call_stats['max_windows_per_frame']}")

    # ----- Bystander analysis summary -----
    if bystander_stats is not None:
        summary = {
            'overall':      dict(bystander_stats['overall']),
            'by_size':      {k: dict(v) for k, v in bystander_stats['by_size'].items()},
            'per_sequence': {k: dict(v) for k, v in bystander_stats['per_sequence'].items()},
        }
        summary_path = os.path.join(args.out_dir, 'aw_bystander_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Console summary
        o = summary['overall']
        total_all = sum(o.values())
        total_oc  = o.get('oc_self', 0) + o.get('oc_bystander', 0)
        print()
        print(f"[AW] bystander summary → {summary_path}")
        print(f"[AW] total detections (post-OBS): {total_all}")
        if total_oc > 0:
            print(f"[AW] from OC windows: {total_oc} ({total_oc / max(1,total_all):.1%} of all)")
            print(f"[AW]   self-detections of tracked object : {o.get('oc_self',0):>6d} "
                  f"({o.get('oc_self',0) / total_oc:.1%} of OC)")
            print(f"[AW]   bystander detections of neighbors: {o.get('oc_bystander',0):>6d} "
                  f"({o.get('oc_bystander',0) / total_oc:.1%} of OC)")
        print()
        print(f"[AW] Breakdown by size bucket (Kos 2022 relative thresholds):")
        print(f"  {'bucket':<8} {'total':>7} {'std':>7} {'oc_self':>8} {'oc_byst':>8} "
              f"{'OC%':>6} {'byst% of OC':>13}")
        for bucket, _lo, _hi in _SIZE_BUCKETS:
            s = summary['by_size'].get(bucket, {})
            tot = sum(s.values())
            if tot == 0:
                continue
            oc_self = s.get('oc_self', 0)
            oc_byst = s.get('oc_bystander', 0)
            oc_t    = oc_self + oc_byst
            oc_frac = oc_t / tot if tot else 0
            byst_frac = oc_byst / oc_t if oc_t else 0
            print(f"  {bucket:<8} {tot:>7d} {s.get('standard',0):>7d} "
                  f"{oc_self:>8d} {oc_byst:>8d} {oc_frac:>5.1%} {byst_frac:>12.1%}")
    # --------------------------------------
