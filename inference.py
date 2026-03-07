import argparse
import os
import json
import time

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
    

    # Get dataset
    ds = DirectoryDataset(
        data_root = args.data_root,
        split = args.split,
        flist = args.flist,
        name = args.name,
    )
    seq2images = ds.seq2images

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

    for seq_name, seq_flist in tqdm(seq2images.items()):
        seq_flist = sorted(seq_flist)

        roi_extractor.reset_predictor() # new tracker for each sequence 
        
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

                if len(det_bboxes) > 0:

                    t1 = time.time()
                    det_dataset =  WindowDetectionDataset(img, ds, det_bboxes, detector.input_size)
                    img_det, det_metadata = det_dataset.get_batch()
                    times['det_get_batch'].append(time.time()-t1)


                    t1 = time.time()
                    detections = detector.get_detections(img_det)
                    times['det_infer'].append(time.time()-t1)

                    t1 = time.time()
                    img_det, img_win = detector.postprocess_detections(detections, det_metadata)
                    times['det_postproc'].append(time.time()-t1)

                    t1 = time.time()
                    # Overlapping Box Suppression
                    img_det = overlapping_box_suppression(img_win, img_det, th=args.obs_iou_th)
                    times['obs'].append(time.time()-t1)

                else:
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
                    frame = make_vis(frame, estim_mask, pred_mask, det_bboxes, img_det, detector.config['classes'], detector.config['colors'], args.vis_conf_th)
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

        
    with open(os.path.join(args.out_dir, f'results-{args.split if args.flist is None else args.name}.json'), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
