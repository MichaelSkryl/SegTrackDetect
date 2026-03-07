"""
Temporal Detection Heatmap — accumulates detection confidence over time,
specifically targeting tiny objects that the segmentation branch misses.

This is NOT a neural network. It's a simple exponential moving average
of detection locations, weighted by object size (smaller objects get
higher weight). The heatmap is OR'd with the segmentation mask before
detection window proposal, making the system more likely to maintain
detection windows in regions where tiny objects were recently seen.

This complements:
  - The bottleneck ConvGRU (which helps medium/large objects)
  - The Kalman tracker (which predicts bounding boxes but doesn't boost
    the segmentation mask for tiny objects specifically)

Requires zero training, zero parameters, negligible compute (<0.1ms).
"""

import cv2
import numpy as np
import torch


class TemporalDetectionHeatmap:
    """
    Maintains a decaying heatmap of past detection locations, with
    size-aware weighting that emphasizes tiny objects.

    Args:
        decay (float): Exponential decay per frame. 0.85 means each frame
            retains 85% of previous heatmap. Default: 0.85.
        tiny_threshold (float): Relative size threshold (fraction of image
            area). Objects below this size get boosted. Default: 0.01 (1%).
        boost_strength (float): Maximum heatmap value at a detection
            location. Default: 1.0.
        activation_threshold (float): Minimum heatmap value to produce
            a mask pixel. Default: 0.3.
        dilation_kernel (int): Kernel size for dilating the boost mask,
            to create a margin around tiny detections. Default: 5.
    """

    def __init__(self, decay=0.85, tiny_threshold=0.01, boost_strength=1.0,
                 activation_threshold=0.3, dilation_kernel=5):
        self.decay = decay
        self.tiny_threshold = tiny_threshold
        self.boost_strength = boost_strength
        self.activation_threshold = activation_threshold
        self.dilation_kernel = dilation_kernel
        self.heatmap = None

    def reset(self):
        """Reset the heatmap at the start of each new sequence."""
        self.heatmap = None

    def update(self, detections, orig_shape, mask_shape):
        """
        Update the heatmap with new detections from the current frame.

        Args:
            detections (np.ndarray or None): (N, 5+) array with columns
                [x1, y1, x2, y2, score, ...]. Can be None or empty.
            orig_shape (tuple): (H, W) of the original full-resolution image.
            mask_shape (tuple): (H, W) of the low-resolution ROI mask.
        """
        mask_h, mask_w = mask_shape
        orig_h, orig_w = orig_shape

        # Initialize or decay
        if self.heatmap is None or self.heatmap.shape != (mask_h, mask_w):
            self.heatmap = np.zeros((mask_h, mask_w), dtype=np.float32)
        else:
            self.heatmap *= self.decay

        if detections is None or len(detections) == 0:
            return

        img_area = orig_h * orig_w
        scale_x = mask_w / orig_w
        scale_y = mask_h / orig_h

        for det in detections:
            x1, y1, x2, y2 = det[:4]
            score = det[4] if len(det) > 4 else 1.0

            # Compute relative size of this detection
            det_area = max(1, (x2 - x1) * (y2 - y1))
            relative_size = det_area / img_area

            # Only boost detections smaller than the threshold
            if relative_size > self.tiny_threshold:
                continue

            # Smaller objects get higher weight (inverse proportional)
            size_weight = max(0.1, 1.0 - relative_size / self.tiny_threshold)

            # Map detection coordinates to mask coordinates
            mx1 = max(0, int(x1 * scale_x))
            my1 = max(0, int(y1 * scale_y))
            mx2 = min(mask_w, int(x2 * scale_x) + 1)
            my2 = min(mask_h, int(y2 * scale_y) + 1)

            if mx2 > mx1 and my2 > my1:
                # Use max to avoid overwriting stronger signals
                boost_val = score * size_weight * self.boost_strength
                self.heatmap[my1:my2, mx1:mx2] = np.maximum(
                    self.heatmap[my1:my2, mx1:mx2], boost_val)

    def get_boost_mask(self):
        """
        Convert the heatmap to a binary boost mask.

        Returns:
            torch.Tensor or None: Binary mask (H, W) of dtype float32.
                Returns None if no heatmap exists yet (first frame).
        """
        if self.heatmap is None:
            return None

        # Threshold the heatmap
        mask = (self.heatmap > self.activation_threshold).astype(np.uint8)

        # Dilate to create margin around tiny object locations
        if self.dilation_kernel > 1 and mask.any():
            kernel = np.ones(
                (self.dilation_kernel, self.dilation_kernel), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return torch.from_numpy(mask.astype(np.float32))
