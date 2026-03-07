import cv2
import numpy as np
import torch


from .estimator import Estimator
from .estimator.postunet_temporal_estimator import PostUnetTemporalEstimator
from .estimator.bottleneck_temporal_estimator import BottleneckTemporalEstimator
from .predictor import Predictor
from .windows_proposals import get_roi_bounding_boxes, get_detection_windows, get_object_centric_windows
from .temporal_heatmap import TemporalDetectionHeatmap


class ROIModule:
    """
    ROIModule manages ROI estimation, prediction, fusion, and detection
    window proposals.

    Supports estimator modes:
      - 'none':        Original frozen TorchScript UNet (baseline)
      - 'post_unet':   ConvGRU after the frozen UNet output
      - 'bottleneck':  ConvGRU inside the UNet at a configurable layer
      - 'dual':        Two ConvGRUs (deep bottleneck + shallow skip)
      - 'postdecoder': ConvGRU after the decoder at full resolution

    Additionally supports:
      - Temporal detection heatmap for tiny object ROI persistence
      - Object-centric adaptive windowing for tiny tracked objects

    Args:
        tracker_name (str): Tracker model name.
        estimator_name (str): Estimator model name (e.g., 'SDS_large').
        is_sequence (bool): Whether input is a video sequence.
        device (str): 'cuda' or 'cpu'.
        bbox_type (str): Detection window filtering method.
        allow_resize (bool): Allow resizing of detection windows.
        use_temporal (str): Temporal mode.
        temporal_hidden (int): Hidden channels for ConvGRU.
        temporal_ks (int): Kernel size for ConvGRU.
        temporal_weights (str or None): Path to trained ConvGRU weights.
        insertion_point (int): Encoder layer for ConvGRU insertion.
        use_heatmap (bool): Whether to use the temporal detection heatmap.
        heatmap_decay (float): Heatmap decay per frame.
        heatmap_tiny_threshold (float): Relative size threshold for heatmap.
        heatmap_activation_threshold (float): Activation threshold for heatmap.
        shallow_hidden (int): Hidden channels for shallow GRU (dual mode).
        shallow_point (int): Encoder layer for shallow GRU (dual mode).
        use_adaptive_windowing (bool): Enable object-centric windows for tiny objects.
        aw_tiny_threshold (float): Relative size threshold for adaptive windowing.
        aw_min_window_ratio (float): Minimum window size as fraction of det_shape.
        aw_padding_factor (float): Padding multiplier around tracked object.
        aw_max_extra_windows (int): Maximum extra windows per frame (FPS cap).
    """

    def __init__(self, tracker_name, estimator_name, is_sequence=True, device='cuda',
                 bbox_type='sorted', allow_resize=True,
                 use_temporal='none', temporal_hidden=16, temporal_ks=3,
                 temporal_weights=None, insertion_point=5,
                 use_heatmap=False, heatmap_decay=0.85,
                 heatmap_tiny_threshold=0.01, heatmap_activation_threshold=0.3,
                 shallow_hidden=16, shallow_point=2,
                 use_adaptive_windowing=False, aw_tiny_threshold=0.01,
                 aw_min_window_ratio=0.5, aw_padding_factor=2.0,
                 aw_max_extra_windows=5):

        self.tracker_name = tracker_name
        self.predictor = Predictor(tracker_name) if is_sequence else None
        self.is_sequence = is_sequence
        self.bbox_type = bbox_type
        self.allow_resize = allow_resize
        self.use_temporal = use_temporal
        self.use_heatmap = use_heatmap

        # Adaptive windowing (object-centric windows for tiny objects)
        self.use_adaptive_windowing = use_adaptive_windowing
        self.aw_tiny_threshold = aw_tiny_threshold
        self.aw_min_window_ratio = aw_min_window_ratio
        self.aw_padding_factor = aw_padding_factor
        self.aw_max_extra_windows = aw_max_extra_windows

        # Temporal detection heatmap
        if use_heatmap:
            self.temporal_heatmap = TemporalDetectionHeatmap(
                decay=heatmap_decay,
                tiny_threshold=heatmap_tiny_threshold,
                activation_threshold=heatmap_activation_threshold,
            )
        else:
            self.temporal_heatmap = None

        # Select estimator based on temporal mode
        if use_temporal == 'bottleneck':
            self.estimator = BottleneckTemporalEstimator(
                estimator_name, device=device,
                gru_hidden=temporal_hidden,
                gru_kernel=temporal_ks,
                gru_weights=temporal_weights,
                insertion_point=insertion_point,
            )
        elif use_temporal == 'post_unet':
            self.estimator = PostUnetTemporalEstimator(
                estimator_name, device=device,
                gru_hidden=temporal_hidden,
                gru_kernel=temporal_ks,
                gru_weights=temporal_weights,
            )
        else:
            self.estimator = Estimator(estimator_name, device=device)


    def get_fused_roi(self, frame_id, img_tensor, orig_shape, det_shape):
        """
        Computes the fused ROI from estimated mask, predicted mask, and
        optional temporal heatmap, then generates detection windows.

        If adaptive windowing is enabled, also generates object-centric
        windows for tiny tracked objects and merges them with standard
        windows.
        """
        self.estimated_mask = self.estimator.get_estimated_roi(
            img_tensor, orig_shape
        )
        estimated_shape = self.estimated_mask.shape[-2:]

        if self.is_sequence:
            self.predicted_mask = self.predictor.get_predicted_roi(
                frame_id, orig_shape, estimated_shape
            )
        else:
            self.predicted_mask = torch.zeros(
                self.estimated_mask.shape, dtype=torch.float32
            )

        fused_mask = torch.logical_or(
            self.estimated_mask.cpu(), self.predicted_mask
        ).float()

        # Add temporal heatmap boost for tiny objects
        if self.temporal_heatmap is not None:
            boost_mask = self.temporal_heatmap.get_boost_mask()
            if boost_mask is not None:
                if boost_mask.shape != fused_mask.shape:
                    boost_mask = torch.from_numpy(
                        cv2.resize(
                            boost_mask.numpy(),
                            (int(fused_mask.shape[-1]), int(fused_mask.shape[-2])),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
                fused_mask = torch.logical_or(
                    fused_mask.bool(), boost_mask.bool()
                ).float()

        fused_mask = (fused_mask.numpy() * 255).astype(np.uint8)
        fused_bboxes = get_roi_bounding_boxes(
            fused_mask, orig_shape, estimated_shape
        )

        # Standard detection windows from ROI blobs
        detection_windows = get_detection_windows(
            fused_bboxes,
            img_shape=orig_shape,
            det_shape=det_shape,
            bbox_type=self.bbox_type,
            allow_resize=self.allow_resize,
        )

        # Adaptive windowing: add object-centric windows for tiny tracked objects
        if self.use_adaptive_windowing and self.is_sequence and frame_id > 0:
            detection_windows = self._merge_adaptive_windows(
                detection_windows, orig_shape, det_shape
            )

        return detection_windows


    def _merge_adaptive_windows(self, standard_windows, orig_shape, det_shape):
        """Generate object-centric windows for tiny tracked objects and merge
        them with the standard windows.

        Object-centric windows are only added if they don't significantly
        overlap with existing standard windows. This ensures we add resolution
        where it's needed without redundant detector runs.

        Args:
            standard_windows (np.ndarray): (N, 4) standard detection windows.
            orig_shape (tuple): (H, W) original image dimensions.
            det_shape (tuple): (H_det, W_det) detector input dimensions.

        Returns:
            np.ndarray: Merged detection windows.
        """
        # Use tracker's predicted bounding boxes as object locations
        if self.predictor is None:
            return standard_windows

        tracked = self.predictor.predicted_bboxes
        if tracked is None or len(tracked) == 0:
            return standard_windows

        tracked_bboxes = np.array(tracked)[:, :4]  # (N, 4) x1,y1,x2,y2

        # Generate object-centric windows for tiny objects
        oc_windows = get_object_centric_windows(
            tracked_bboxes,
            img_shape=orig_shape,
            det_shape=det_shape,
            tiny_threshold=self.aw_tiny_threshold,
            min_window_ratio=self.aw_min_window_ratio,
            padding_factor=self.aw_padding_factor,
        )

        if len(oc_windows) == 0:
            return standard_windows

        # Filter: remove object-centric windows that are already well-covered
        # by standard windows (if >70% of the oc_window area is inside a
        # standard window, the standard window already covers it)
        novel_windows = []
        for oc_win in oc_windows:
            if not self._is_well_covered(oc_win, standard_windows, coverage_threshold=0.7):
                novel_windows.append(oc_win)

        if not novel_windows:
            return standard_windows

        # Cap the number of extra windows to control FPS impact
        novel_windows = novel_windows[:self.aw_max_extra_windows]
        novel_windows = np.array(novel_windows, dtype=np.int32)

        # Merge
        if len(standard_windows) == 0:
            return novel_windows
        return np.concatenate([standard_windows, novel_windows], axis=0)


    @staticmethod
    def _is_well_covered(window, existing_windows, coverage_threshold=0.7):
        """Check if a window is already well-covered by any existing window.

        Args:
            window (array-like): [x1, y1, x2, y2] of the candidate window.
            existing_windows (np.ndarray): (N, 4) existing windows.
            coverage_threshold (float): Fraction of window area that must
                be inside an existing window to count as "covered".

        Returns:
            bool: True if the window is well-covered.
        """
        if len(existing_windows) == 0:
            return False

        wx1, wy1, wx2, wy2 = window
        win_area = max(1, (wx2 - wx1) * (wy2 - wy1))

        for ew in existing_windows:
            ex1, ey1, ex2, ey2 = ew

            # Intersection
            ix1 = max(wx1, ex1)
            iy1 = max(wy1, ey1)
            ix2 = min(wx2, ex2)
            iy2 = min(wy2, ey2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter_area = (ix2 - ix1) * (iy2 - iy1)

            # What fraction of the candidate window is covered?
            coverage = inter_area / win_area

            # Also check: is the existing window LARGER? If the candidate
            # is smaller (tighter crop), it provides better resolution
            # even if covered. Only skip if existing window is same size
            # or smaller (meaning it already provides good resolution).
            ew_area = (ex2 - ex1) * (ey2 - ey1)

            # If the existing window is at most 1.5× the candidate size
            # and covers >70% of it, the candidate is redundant
            if coverage > coverage_threshold and ew_area <= win_area * 1.5:
                return True

        return False


    def reset_predictor(self):
        """Reset tracker, temporal state, and heatmap for a new sequence."""
        if self.predictor is not None:
            self.predictor = Predictor(self.tracker_name)
        if self.use_temporal in ('post_unet', 'bottleneck', 'dual', 'postdecoder'):
            self.estimator.reset_temporal_state()
        if self.temporal_heatmap is not None:
            self.temporal_heatmap.reset()

    def update_predictor(self, detections):
        """Update tracker state with new detections."""
        if self.predictor is not None:
            self.predictor.update_tracker_state(detections)

    def update_heatmap(self, detections, orig_shape):
        """
        Update the temporal detection heatmap with current frame detections.

        Args:
            detections (np.ndarray): (N, 6) array [x1, y1, x2, y2, score, class].
            orig_shape (tuple): (H, W) of the original image.
        """
        if self.temporal_heatmap is not None:
            mask_shape = self.estimated_mask.shape[-2:]
            self.temporal_heatmap.update(
                detections=detections,
                orig_shape=orig_shape,
                mask_shape=mask_shape,
            )

    def get_masks(self, orig_shape):
        """Get estimated and predicted masks resized to original shape."""
        estim = self.estimated_mask.cpu().numpy()
        pred = self.predicted_mask.cpu().numpy()
        estim = cv2.resize(estim, (orig_shape[1], orig_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        return estim, pred


    def get_config_dict(self):
        """Return a serializable config dictionary."""
        config = {
            'tracker_name': self.tracker_name,
            'bbox_type': self.bbox_type,
            'allow_resize': self.allow_resize,
            'use_temporal': self.use_temporal,
            'estimator_input_size': list(self.estimator.input_size),
            'use_heatmap': self.use_heatmap,
            'use_adaptive_windowing': self.use_adaptive_windowing,
        }
        if self.temporal_heatmap is not None:
            config['heatmap_decay'] = self.temporal_heatmap.decay
            config['heatmap_tiny_threshold'] = self.temporal_heatmap.tiny_threshold
            config['heatmap_activation_threshold'] = self.temporal_heatmap.activation_threshold
        if self.use_adaptive_windowing:
            config['aw_tiny_threshold'] = self.aw_tiny_threshold
            config['aw_min_window_ratio'] = self.aw_min_window_ratio
            config['aw_padding_factor'] = self.aw_padding_factor
            config['aw_max_extra_windows'] = self.aw_max_extra_windows
        return config
