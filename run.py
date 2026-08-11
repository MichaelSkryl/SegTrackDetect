#!/usr/bin/env python3
"""
run.py — Run the extended SegTrackDetect pipeline on arbitrary media.

Unlike `inference.py`, this script does **not** require a COCO-annotated
dataset. It accepts a single image, a video file, a directory of frames, or a
webcam index, and produces annotated output plus optional JSON detections.

Place this file in the repository root (next to `inference.py`).

Examples
--------
Single image:
    python run.py --source input/photo.jpg \\
        --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 \\
        --out_dir output/photo

Video with all modifications enabled:
    python run.py --source input/clip.mp4 \\
        --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 \\
        --allow_resize --use_adaptive_windowing \\
        --out_dir output/clip

Directory of ordered frames (treated as one video sequence):
    python run.py --source input/frames/ \\
        --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 \\
        --out_dir output/frames

Webcam, live window:
    python run.py --source 0 --show \\
        --roi_model Airport_tiny_batch_8 --det_model AirportYolov7
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms as T

from datasets import WindowDetectionDataset
from drawing import make_vis
from rois import ROIModule
from detector import Detector, overlapping_box_suppression
from detector.aggregation import xyxy2xywh


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.mpg', '.mpeg', '.wmv'}


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class FrameSource:
    """Unified iterator over an image, a video, a directory or a camera.

    Yields ``(frame_bgr, name)`` pairs. Exposes ``kind``, ``fps``, ``n_frames``
    and ``is_sequence`` so the caller can configure the tracker correctly:
    a lone image has no temporal context, everything else does.
    """

    def __init__(self, source, stride=1, max_frames=None):
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.cap = None
        self.paths = None

        if isinstance(source, str) and source.isdigit():
            self.kind = 'camera'
            self.cap = cv2.VideoCapture(int(source))
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open camera index {source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.n_frames = None
            self.name = f"camera{source}"

        elif os.path.isdir(source):
            self.kind = 'folder'
            files = [f for f in sorted(os.listdir(source))
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            if not files:
                raise SystemExit(f"No images found in directory: {source}")
            self.paths = [os.path.join(source, f) for f in files]
            self.fps = 25.0
            self.n_frames = len(self.paths)
            self.name = os.path.basename(os.path.normpath(source))

        elif os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext in IMAGE_EXTS:
                self.kind = 'image'
                self.paths = [source]
                self.fps = 1.0
                self.n_frames = 1
            elif ext in VIDEO_EXTS:
                self.kind = 'video'
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    raise SystemExit(f"Cannot open video: {source}")
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.n_frames = total if total > 0 else None
            else:
                raise SystemExit(
                    f"Unsupported file extension '{ext}'.\n"
                    f"Images: {sorted(IMAGE_EXTS)}\nVideos: {sorted(VIDEO_EXTS)}")
            self.name = os.path.splitext(os.path.basename(source))[0]

        else:
            raise SystemExit(f"Source not found: {source}")

        # A single still image carries no temporal information, so the tracker
        # and every temporal module must be disabled for it.
        self.is_sequence = self.kind != 'image'

    def __iter__(self):
        emitted = 0
        if self.paths is not None:
            for i, p in enumerate(self.paths):
                if i % self.stride:
                    continue
                frame = cv2.imread(p)
                if frame is None:
                    print(f"  [warn] unreadable, skipping: {p}")
                    continue
                yield frame, os.path.basename(p)
                emitted += 1
                if self.max_frames and emitted >= self.max_frames:
                    return
        else:
            i = 0
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                if i % self.stride == 0:
                    yield frame, f"{i:06d}.jpg"
                    emitted += 1
                    if self.max_frames and emitted >= self.max_frames:
                        break
                i += 1
            self.cap.release()

    def expected_frames(self):
        if self.n_frames is None:
            return None
        n = (self.n_frames + self.stride - 1) // self.stride
        return min(n, self.max_frames) if self.max_frames else n


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Run extended SegTrackDetect on images, video or a camera.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- source / output -------------------------------------------------
    g = p.add_argument_group('input / output')
    g.add_argument('--source', required=True,
                   help='Image file, video file, directory of frames, or camera index.')
    g.add_argument('--out_dir', default='output/run',
                   help='Directory for annotated media and detections.')
    g.add_argument('--show', action='store_true',
                   help='Display a live window (needs a display; see README for Docker).')
    g.add_argument('--no_save', action='store_true',
                   help='Do not write annotated media to disk.')
    g.add_argument('--save_json', action='store_true',
                   help='Also write detections.json (COCO-style boxes, xywh).')
    g.add_argument('--stride', type=int, default=1,
                   help='Process every Nth frame.')
    g.add_argument('--max_frames', type=int, default=None,
                   help='Stop after this many processed frames.')
    g.add_argument('--vis_scale', type=float, default=1.0,
                   help='Scale factor for the rendered output (0.5 halves 4K).')

    # --- models ----------------------------------------------------------
    g = p.add_argument_group('models')
    g.add_argument('--roi_model', default='Airport_tiny_batch_8',
                   help='Key in rois.estimator.configs.ESTIMATOR_MODELS.')
    g.add_argument('--det_model', default='AirportYolov7',
                   help='Key in detector.configs.DETECTION_MODELS.')
    g.add_argument('--tracker', default='sort',
                   help='Key in rois.predictor.configs.PREDICTOR_MODELS.')
    g.add_argument('--cpu', action='store_true', help='Force CPU inference.')

    # --- windowing -------------------------------------------------------
    g = p.add_argument_group('detection windows')
    g.add_argument('--bbox_type', default='sorted',
                   choices=['all', 'naive', 'sorted'],
                   help='Window filtering strategy.')
    g.add_argument('--allow_resize', action='store_true',
                   help='Downscale oversized windows instead of sliding-window tiling.')
    g.add_argument('--obs_iou_th', type=float, default=0.7,
                   help='IoU threshold for Overlapping Box Suppression.')

    # --- modification 1: ConvGRU ----------------------------------------
    g = p.add_argument_group('modification: temporal ConvGRU')
    g.add_argument('--use_temporal', default='none',
                   choices=['none', 'bottleneck'],
                   help="'bottleneck' enables the ConvGRU in the ROI encoder.")
    g.add_argument('--temporal_weights', default=None,
                   help='Path to trained ConvGRU / full-model weights.')
    # Defaults deliberately match inference.py so the same command line works
    # with either entry point. Both values MUST match the checkpoint; look them
    # up in train_args.json next to the weights (gru_hidden / insertion_point).
    g.add_argument('--temporal_hidden', type=int, default=16,
                   help='ConvGRU hidden channels — training calls this --gru_hidden.')
    g.add_argument('--temporal_ks', type=int, default=3,
                   help='ConvGRU kernel size — training calls this --gru_kernel.')
    g.add_argument('--insertion_point', type=int, default=5, choices=[2, 3, 4, 5],
                   help='Encoder layer for ConvGRU insertion (must match checkpoint).')

    # --- modification 2: heatmap ----------------------------------------
    g = p.add_argument_group('modification: detection significance heatmap')
    g.add_argument('--use_heatmap', action='store_true')
    g.add_argument('--heatmap_decay', type=float, default=0.85)
    g.add_argument('--heatmap_tiny_threshold', type=float, default=0.01)
    g.add_argument('--heatmap_activation_threshold', type=float, default=0.3)

    # --- modification 3: adaptive windowing ------------------------------
    g = p.add_argument_group('modification: object-centric adaptive windowing')
    g.add_argument('--use_adaptive_windowing', action='store_true')
    g.add_argument('--aw_tiny_threshold', type=float, default=0.01)
    g.add_argument('--aw_min_window_ratio', type=float, default=0.5)
    g.add_argument('--aw_padding_factor', type=float, default=2.0)
    g.add_argument('--aw_max_extra_windows', type=int, default=5)

    # --- visualisation ---------------------------------------------------
    g = p.add_argument_group('visualisation')
    g.add_argument('--vis_conf_th', type=float, default=0.3,
                   help='Only draw detections above this confidence.')
    g.add_argument('--draw_masks', action='store_true',
                   help='Overlay the ROI masks (blue = segmentation, orange = tracker).')
    g.add_argument('--draw_windows', action='store_true',
                   help='Draw detection windows as black rectangles.')
    g.add_argument('--draw_oc_windows', action='store_true',
                   help='Highlight object-centric windows in yellow.')
    g.add_argument('--no_labels', action='store_true',
                   help='Draw boxes without class/confidence text.')
    g.add_argument('--hud', action='store_true',
                   help='Overlay FPS and per-frame detection/window counts.')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(frame, detections, windows, oc_windows, oc_sources,
           masks, classes, colors, args):
    """Draw the pipeline output onto a BGR frame."""
    estim_mask, pred_mask = masks if args.draw_masks else (None, None)
    std_windows = windows if args.draw_windows else np.empty((0, 4), np.int32)

    frame = make_vis(frame, estim_mask, pred_mask, std_windows, detections,
                     classes, colors, args.vis_conf_th,
                     show_label=not args.no_labels)

    if args.draw_oc_windows and oc_windows is not None and len(oc_windows):
        for w, s in zip(oc_windows, oc_sources):
            cv2.rectangle(frame, (int(w[0]), int(w[1])),
                          (int(w[2]), int(w[3])), (0, 255, 255), 3)
            cv2.rectangle(frame, (int(s[0]), int(s[1])),
                          (int(s[2]), int(s[3])), (255, 0, 255), 2)
            cv2.putText(frame, 'OC', (int(w[0]) + 4, int(w[1]) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def draw_hud(frame, fps, n_det, n_win):
    txt = f"{fps:5.1f} FPS | {n_det} det | {n_win} win"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (8, 8), (18 + tw, 22 + th), (0, 0, 0), -1)
    cv2.putText(frame, txt, (13, 16 + th), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    source = FrameSource(args.source, stride=args.stride,
                         max_frames=args.max_frames)
    print(f"Source     : {args.source}  ({source.kind}, "
          f"{source.expected_frames() or 'unknown'} frames to process)")

    # A single image cannot use any temporal module — say so rather than
    # silently producing different behaviour than the flags suggest.
    if not source.is_sequence:
        disabled = [n for n, on in (('--use_temporal', args.use_temporal != 'none'),
                                    ('--use_heatmap', args.use_heatmap),
                                    ('--use_adaptive_windowing',
                                     args.use_adaptive_windowing)) if on]
        if disabled:
            print(f"  [note] single image: {', '.join(disabled)} have no effect "
                  f"(no temporal context is available).")

    # The estimator skips weight loading when the path does not exist, which
    # leaves an untrained near-identity ConvGRU in place and quietly reproduces
    # baseline results. Fail here instead.
    if args.use_temporal != 'none':
        if not args.temporal_weights:
            raise SystemExit(
                "--use_temporal bottleneck requires --temporal_weights. "
                "Without a checkpoint the ConvGRU is untrained and the run is "
                "equivalent to --use_temporal none.")
        if not os.path.isfile(args.temporal_weights):
            raise SystemExit(f"--temporal_weights not found: {args.temporal_weights}")
        print(f"ConvGRU    : {args.temporal_weights} "
              f"(insertion_point={args.insertion_point}, "
              f"hidden={args.temporal_hidden}, kernel={args.temporal_ks})")

    device = torch.device('cuda:0') if (torch.cuda.is_available() and not args.cpu) \
        else torch.device('cpu')
    print(f"Device     : {device}")

    detector = Detector(args.det_model, device)
    roi_extractor = ROIModule(
        tracker_name=args.tracker,
        estimator_name=args.roi_model,
        is_sequence=source.is_sequence,
        device=device,
        bbox_type=args.bbox_type,
        allow_resize=args.allow_resize,
        use_temporal=args.use_temporal,
        temporal_hidden=args.temporal_hidden,
        temporal_ks=args.temporal_ks,
        temporal_weights=args.temporal_weights,
        insertion_point=args.insertion_point,
        use_heatmap=args.use_heatmap,
        heatmap_decay=args.heatmap_decay,
        heatmap_tiny_threshold=args.heatmap_tiny_threshold,
        heatmap_activation_threshold=args.heatmap_activation_threshold,
        use_adaptive_windowing=args.use_adaptive_windowing,
        aw_tiny_threshold=args.aw_tiny_threshold,
        aw_min_window_ratio=args.aw_min_window_ratio,
        aw_padding_factor=args.aw_padding_factor,
        aw_max_extra_windows=args.aw_max_extra_windows,
    )
    roi_extractor.aw_verbose = False          # keep the console clean
    roi_extractor.reset_predictor()

    roi_transform = roi_extractor.estimator.preprocess(
        **roi_extractor.estimator.preprocess_args)
    classes = detector.config['classes']
    colors = detector.config['colors']

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    writer = None
    frames_dir = None
    if not args.no_save:
        if source.kind in ('video', 'camera'):
            os.makedirs(args.out_dir, exist_ok=True)      # video written lazily
        else:
            frames_dir = os.path.join(args.out_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

    annotations = []
    to_tensor = T.ToTensor()
    fps_window = deque(maxlen=30)
    n_processed = 0
    t_start = time.time()

    print("Running... (Ctrl+C to stop)\n")
    try:
        with torch.inference_mode():
            for frame_id, (frame_bgr, frame_name) in enumerate(source):
                t0 = time.time()
                H, W = frame_bgr.shape[:2]

                # Prepare tensors exactly as ROIDataset does in inference.py
                rgb = cv2.cvtColor(np.ascontiguousarray(frame_bgr),
                                   cv2.COLOR_BGR2RGB)
                img = to_tensor(rgb).half().unsqueeze(0).to(device)
                img_roi = roi_transform(img)

                # 1) region proposals -> detection windows
                det_bboxes = roi_extractor.get_fused_roi(
                    frame_id=frame_id, img_tensor=img_roi,
                    orig_shape=(H, W), det_shape=detector.input_size)

                # 2) detection inside the windows
                if len(det_bboxes) > 0:
                    win_ds = WindowDetectionDataset(
                        img, None, det_bboxes, detector.input_size)
                    batch, meta = win_ds.get_batch()
                    raw = detector.get_detections(batch)
                    img_det, img_win = detector.postprocess_detections(raw, meta)
                    img_det = overlapping_box_suppression(
                        img_win, img_det, th=args.obs_iou_th)
                else:
                    img_det = torch.empty((0, 6), device=device)

                det_np = img_det.detach().cpu().numpy()

                # 3) feed the temporal modules for the next frame
                if args.use_heatmap:
                    roi_extractor.update_heatmap(detections=det_np,
                                                 orig_shape=(H, W))
                roi_extractor.update_predictor(det_np[:, :-1])

                # 4) render
                masks = roi_extractor.get_masks((H, W)) if args.draw_masks \
                    else (None, None)
                vis = render(frame_bgr.copy(), img_det, det_bboxes,
                             getattr(roi_extractor, 'last_oc_windows', None),
                             getattr(roi_extractor, 'last_oc_sources', None),
                             masks, classes, colors, args)

                dt = time.time() - t0
                fps_window.append(dt)
                fps = len(fps_window) / max(sum(fps_window), 1e-6)
                if args.hud:
                    vis = draw_hud(vis, fps, len(det_np), len(det_bboxes))

                if args.vis_scale != 1.0:
                    vis = cv2.resize(
                        vis, (int(vis.shape[1] * args.vis_scale),
                              int(vis.shape[0] * args.vis_scale)),
                        interpolation=cv2.INTER_AREA)

                # 5) persist
                if not args.no_save:
                    if source.kind in ('video', 'camera'):
                        if writer is None:
                            out_path = os.path.join(
                                args.out_dir, f"{source.name}_annotated.mp4")
                            writer = cv2.VideoWriter(
                                out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                source.fps / args.stride,
                                (vis.shape[1], vis.shape[0]))
                            if not writer.isOpened():
                                raise SystemExit(
                                    f"Could not open video writer for {out_path}. "
                                    f"Is ffmpeg available in the container?")
                            print(f"Writing    : {out_path}")
                        writer.write(vis)
                    else:
                        cv2.imwrite(os.path.join(frames_dir, frame_name), vis)

                if args.save_json and len(det_np):
                    xywh = xyxy2xywh(det_np[:, :4].copy())
                    for row, box in zip(det_np, xywh):
                        annotations.append({
                            'id': len(annotations),
                            'frame': frame_id,
                            'file_name': frame_name,
                            'category_id': int(row[5]),
                            'category': classes[int(row[5])],
                            'bbox': [round(float(v), 2) for v in box],
                            'score': round(float(row[4]), 5),
                        })

                if args.show:
                    cv2.imshow('SegTrackDetect', vis)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        print("\nStopped by user.")
                        break

                n_processed += 1
                if n_processed % 10 == 0 or source.kind == 'image':
                    total = source.expected_frames()
                    pos = f"{n_processed}/{total}" if total else str(n_processed)
                    print(f"  frame {pos:>12} | {fps:5.1f} FPS | "
                          f"{len(det_np):3d} det | {len(det_bboxes):2d} win",
                          end='\r', flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n\nProcessed {n_processed} frames in {elapsed:.1f} s "
          f"({n_processed / max(elapsed, 1e-6):.2f} FPS average)")

    if args.save_json:
        json_path = os.path.join(args.out_dir, 'detections.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        print(f"Detections : {json_path}  ({len(annotations)} boxes)")

    if not args.no_save:
        print(f"Output     : {os.path.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
