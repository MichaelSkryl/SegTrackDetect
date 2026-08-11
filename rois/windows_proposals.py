import math
import time
import logging

import cv2
import numpy as np

import torch
import kornia
from torchvision import transforms as T
from collections import OrderedDict

import torch.nn.functional as F
from torchvision.ops import boxes


# Module-level logger for adaptive-windowing diagnostics.
# Silent by default (NullHandler). inference.py attaches a FileHandler so
# the messages land in a log file instead of the console.
_aw_log = logging.getLogger('adaptive_windowing')
_aw_log.addHandler(logging.NullHandler())




def get_roi_bounding_boxes(binary_mask, original_shape, current_shape):
    """Extracts ROI bounding boxes from a binary mask and rescales them to the original image shape.

    Args:
        binary_mask (np.ndarray): Binary mask from which contours are extracted.
        original_shape (tuple): Original image shape as (height, width).
        current_shape (tuple): Current shape of the mask (height, width).

    Returns:
        np.ndarray: Array of bounding boxes with shape (N, 4) where each box is defined as (xmin, ymin, xmax, ymax).
    """
    orig_height, orig_width = original_shape
    curr_height, curr_width = current_shape

    scale_x = orig_width / curr_width
    scale_y = orig_height / curr_height

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return empty array if no contours
    if not contours:
        return np.empty((0, 4))

    # Get bounding boxes for all contours
    bboxes = np.array([cv2.boundingRect(contour) for contour in contours])

    # Unpack the bounding box data
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Scale bounding boxes
    xmin = np.maximum(0, (x * scale_x))
    ymin = np.maximum(0, (y * scale_y))
    xmax = np.minimum(orig_width, ((x + w) * scale_x))
    ymax = np.minimum(orig_height, ((y + h) * scale_y))

    # Stack into (N, 4) array
    bounding_boxes = np.stack([xmin, ymin, xmax, ymax], axis=1)

    return bounding_boxes.astype(np.int32)



def rotate(bbox_roi, det_shape):
    """Checks if an ROI needs to be rotated after cropping to fit the orientation of the detection window.

    Args:
        bbox_roi (list): Bounding box for the ROI [xmin, ymin, xmax, ymax].
        det_shape (tuple): Shape of the detection window (height, width).

    Returns:
        bool: True if rotation is needed, False otherwise.
    """
    def same_orientation(h_roi, w_roi, h_window, w_window):
        if h_roi >= w_roi and h_window >= w_window:
            return True
        if h_roi < w_roi and h_window < w_window:
            return True
        return False

    h_window, w_window = det_shape
    xmin, ymin, xmax, ymax = bbox_roi
    h_roi, w_roi = ymax-ymin, xmax-xmin
    
    if same_orientation(h_roi, w_roi, h_window, w_window):
        return False
    return True



def get_sliding_windows(roi_bbox, img_shape, det_shape, overlap_px = 20):
    """Generates sliding windows within a given ROI.

    Args:
        roi_bbox (list): Bounding box for the ROI [xmin, ymin, xmax, ymax].
        img_shape (tuple): Shape of the image (height, width).
        det_shape (tuple): Shape of the detection window (height, width).
        overlap_px (int, optional): Overlap between sliding windows. Defaults to 20.

    Returns:
        tuple: List of bounding boxes for sliding windows and a full bounding box covering the entire ROI.
    """
    h,w = det_shape

    xmin, ymin, xmax, ymax = roi_bbox
    roi_h, roi_w = ymax-ymin, xmax-xmin

    n_x = math.ceil(roi_w / (w-overlap_px))
    n_y = math.ceil(roi_h / (h-overlap_px))

    XS = np.array([xmin+(w-overlap_px)*n for n in range(n_x)])
    YS = np.array([ymin+(h-overlap_px)*n for n in range(n_y)])

    # max_xmax = XS[-1]+w
    # max_ymax = YS[-1]+h

    bboxes = []
    for _xmin in XS:
        for _ymin in YS:
            _xmin,_ymin = [int(_) for _ in [_xmin,_ymin]]
            _xmax,_ymax = min(_xmin+w, img_shape[1]), min(_ymin+h, img_shape[0])
            # _xmax,_ymax = min(_xmin+w, xmax), min(_ymin+h, ymax)
            bboxes.append([_xmin, _ymin, _xmax, _ymax])
    full_bbox = [min([x[0] for x in bboxes]), min([x[1] for x in bboxes]), max([x[2] for x in bboxes]),  max([x[3] for x in bboxes])]
    return bboxes, full_bbox



def get_detection_window(roi_bbox, img_shape, det_shape, padding=20, allow_resize=False):
    """Generates a detection window for an ROI.

    Args:
        roi_bbox (list): Bounding box for the ROI [xmin, ymin, xmax, ymax].
        img_shape (tuple): Shape of the image (height, width).
        det_shape (tuple): Shape of the detection window (height, width).
        padding (int, optional): Padding to apply around the ROI. Defaults to 20.
        allow_resize (bool, optional): Whether to allow resizing of the ROI. Defaults to False.

    Returns:
        tuple: Lists of crop bounding boxes and full bounding boxes.
    """
    rot = rotate(roi_bbox, det_shape)
    if rot:
        w,h = det_shape
    else:
        h,w = det_shape
    ar = h/w 
        
    xmin, ymin, xmax, ymax = roi_bbox
    h_roi, w_roi = ymax-ymin, xmax-xmin

    # sliding-window within ROI region
    needs_sliding = (h_roi > h or w_roi > w) and not allow_resize
    if needs_sliding:
        crop_bboxes, full_bbox = get_sliding_windows(roi_bbox, img_shape, det_shape) # check if empty
        return [crop_bboxes], [full_bbox]

    # Apply padding if needed
    h_roi = max(h_roi + padding, h)
    w_roi = max(w_roi + padding, w)

    # Adjust ROI size to maintain aspect ratio
    if h_roi / w_roi != ar:
        if h_roi / w_roi > ar:  # ROI is taller, adjust width
            w_roi = h_roi / ar
        else:  # ROI is wider, adjust height
            h_roi = w_roi * ar

    xc = (xmax+xmin)//2
    yc = (ymin+ymax)//2

    xmin = max(xc - w_roi//2, 0)
    ymin = max(yc - h_roi//2, 0)

    xmax = xmin+w_roi
    ymax = ymin+h_roi

    if xmax > img_shape[1]:
        dx = xmax-img_shape[1]
        xmin = max(xmin - dx, 0)
        xmax = max(xmax - dx, 0)
    if ymax > img_shape[0]:
        dy = ymax-img_shape[0]
        ymin = max(ymin - dy, 0)
        ymax = max(ymax - dy, 0)

    crop_bbox = [xmin, ymin, xmax, ymax]
    return [crop_bbox], [crop_bbox]



def get_detection_windows(rois, img_shape, det_shape=(960, 960), bbox_type='naive', allow_resize=True):
    """Processes a set of ROIs and generates detection windows.

    Args:
        rois (np.ndarray): Array of ROI bounding boxes (N, 4).
        img_shape (tuple): Shape of the image (height, width).
        det_shape (tuple, optional): Shape of the detection window (height, width). Defaults to (960, 960).
        bbox_type (str, optional): Type of bounding box filtering ('naive', 'sorted', 'all'). Defaults to 'naive'.
        allow_resize (bool, optional): Whether to allow resizing of detection windows. Defaults to True.

    Returns:
        np.ndarray: Array of detection windows with shape (N, 4) where each window is defined as (xmin, ymin, xmax, ymax).
    """
    crop_windows, full_windows = [], []
    for roi in rois:
        crop, full = get_detection_window(roi, img_shape, det_shape=det_shape, allow_resize=allow_resize)
        crop_windows+=crop
        full_windows+=full

    if bbox_type == 'all':
        detection_windows = crop_windows
    elif bbox_type == 'naive':
        detection_windows = filter_detection_windows_naive(rois, crop_windows, full_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
    elif bbox_type == 'sorted':
        detection_windows =  filter_detection_windows_sorted(rois, crop_windows, full_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)
    else:
        raise NotImplementedError

    detection_windows = np.array(detection_windows).astype(np.int32)
    if len(detection_windows) > 0:
        detection_windows[:, 0] = np.clip(detection_windows[:, 0], 0, img_shape[1])
        detection_windows[:, 2] = np.clip(detection_windows[:, 2], 0, img_shape[1])
        detection_windows[:, 1] = np.clip(detection_windows[:, 1], 0, img_shape[0])
        detection_windows[:, 3] = np.clip(detection_windows[:, 3], 0, img_shape[0])
        indices = np.nonzero(((detection_windows[:,2]-detection_windows[:,0]) > 0) & ((detection_windows[:,3]-detection_windows[:,1]) > 0))
        detection_windows = detection_windows[indices[0], :]
    return detection_windows


    
def is_bbox_inside_bbox(inner_bbox, outer_bbox):
    """Checks if an inner bounding box is fully inside an outer bounding box.

    Args:
        inner_bbox (list or tuple): Bounding box in the format [xmin, ymin, xmax, ymax].
        outer_bbox (list or tuple): Bounding box in the format [xmin, ymin, xmax, ymax].

    Returns:
        bool: True if all four corners of the inner bounding box are within the outer bounding box, False otherwise.
    """
    def is_point_inside_bbox(point, bbox):
        """Checks if a given point is inside a bounding box.

        Args:
            point (tuple): A point (x, y) to check.
            bbox (list or tuple): Bounding box in the format [xmin, ymin, xmax, ymax].

        Returns:
            bool: True if the point is inside the bounding box, False otherwise.
        """
        xmin, ymin, xmax, ymax = bbox
        x, y = point
        if xmin<=x<=xmax and ymin<=y<=ymax:
            return True

    xmin,ymin,xmax,ymax = inner_bbox
    p1, p2, p3, p4 = (xmin,ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)
    return all([is_point_inside_bbox(point, outer_bbox) for point in [p1,p2,p3,p4]])



def filter_detection_windows_naive(rois, crop_windows, full_windows, img_shape, det_shape, allow_resize):
    """Filters detection windows by removing redundant windows where ROIs are already covered by previous windows.

    Args:
        rois (list of list): List of ROIs, each represented as a bounding box [xmin, ymin, xmax, ymax].
        crop_windows (list of list): List of crop windows corresponding to the ROIs.
        full_windows (list of list): List of full windows corresponding to the ROIs.
        img_shape (tuple): Shape of the input image (height, width).
        det_shape (tuple): Shape of the detection window (height, width).
        allow_resize (bool): Flag indicating whether resizing is allowed for sliding windows.

    Returns:
        list of list: The final list of filtered crop windows, without redundant windows.
    """
    final_crop_windows, final_full_windows = [],[]
    for i in range(len(rois)):
        if any([is_bbox_inside_bbox(rois[i], final_full_window) for final_full_window in final_full_windows]):
            continue
            
        if isinstance(crop_windows[i][0], list):
            final_crop_windows+=crop_windows[i]
        else:
            final_crop_windows.append(crop_windows[i])
        final_full_windows.append(full_windows[i])

    return final_crop_windows



def filter_detection_windows_sorted(rois, crop_windows, full_windows, img_shape, det_shape, allow_resize):
    """Filters detection windows by sorting based on the number of ROIs inside each window,
    and then removes redundant windows.

    Args:
        rois (list of list): List of ROIs, each represented as a bounding box [xmin, ymin, xmax, ymax].
        crop_windows (list of list): List of crop windows corresponding to the ROIs.
        full_windows (list of list): List of full windows corresponding to the ROIs.
        img_shape (tuple): Shape of the input image (height, width).
        det_shape (tuple): Shape of the detection window (height, width).
        allow_resize (bool): Flag indicating whether resizing is allowed for sliding windows.

    Returns:
        list of list: The final list of filtered crop windows, sorted by the number of ROIs covered.
    """
    hits = {i: 0 for i in range(len(full_windows))}
    for i, full_window in enumerate(full_windows):
        hits[i]+=sum([is_bbox_inside_bbox(roi, full_window) for roi in rois])

    # hits - how many rois inside each window, sort - det_windows with many rois first
    hits = OrderedDict(sorted(hits.items(), key=lambda item: item[1], reverse=True))
    rois = [rois[i] for i in hits.keys()]
    crop_windows = [crop_windows[i] for i in hits.keys()]
    full_windows = [full_windows[i] for i in hits.keys()]

    final_crop_windows = filter_detection_windows_naive(rois, crop_windows, full_windows, img_shape, det_shape=det_shape, allow_resize=allow_resize)

    return final_crop_windows 
    
    
def get_object_centric_windows(tracked_objects, img_shape, det_shape,
                               tiny_threshold=0.01, min_window_ratio=0.5,
                               padding_factor=2.0, return_sources=False,
                               verbose=False):
    """Generate tight detection windows centered on individual tracked tiny objects.

    For each tracked object smaller than `tiny_threshold` (relative to image area),
    creates a detection window tightly centered on the object. The window size is
    proportional to the object size (with padding), but at least `min_window_ratio`
    of the standard detection window. This gives the detector more pixels per
    tiny object compared to large ROI-blob-based windows.

    Args:
        tracked_objects (np.ndarray): (N, 4+) array of tracked bounding boxes
            [x1, y1, x2, y2, ...] in original image coordinates.
        img_shape (tuple): (H, W) of the original image.
        det_shape (tuple): (H_det, W_det) standard detector input size.
        tiny_threshold (float): Relative size threshold. Objects with area /
            image_area < threshold get object-centric windows. Default: 0.01.
        min_window_ratio (float): Minimum window size as fraction of det_shape.
            0.5 means the smallest window is 256×256 for a 512×512 detector.
            Default: 0.5.
        padding_factor (float): How much larger the window is relative to the
            object. 2.0 means the window is 2× the object size in each dim.
            Default: 2.0.
        return_sources (bool): If True, also returns the index of the source
            tracked object for each surviving window.
        verbose (bool): If True, prints per-object reasoning.

    Returns:
        np.ndarray: (M, 4) array of object-centric detection windows.
        np.ndarray (optional): (M,) source indices into tracked_objects.
    """
    if tracked_objects is None or len(tracked_objects) == 0:
        if return_sources:
            return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.int32)
        return np.empty((0, 4), dtype=np.int32)

    img_h, img_w = img_shape
    img_area = img_h * img_w
    det_h, det_w = det_shape

    # Minimum window dimensions
    min_h = int(det_h * min_window_ratio)
    min_w = int(det_w * min_window_ratio)

    windows = []
    sources = []
    for idx, obj in enumerate(tracked_objects):
        x1, y1, x2, y2 = obj[:4]
        obj_w = max(1, x2 - x1)
        obj_h = max(1, y2 - y1)
        obj_area = obj_w * obj_h
        relative_size = obj_area / img_area

        # Only create object-centric windows for tiny objects
        if relative_size >= tiny_threshold:
            if verbose:
                _aw_log.info(f"  [OC] obj[{idx}] area={int(obj_area)} rel={relative_size:.5f} "
                             f">= {tiny_threshold} → SKIP (not tiny)")
            continue

        # Window size: padded object size, but at least min_window
        win_w = max(int(obj_w * padding_factor), min_w)
        win_h = max(int(obj_h * padding_factor), min_h)

        # Maintain detector aspect ratio
        det_ar = det_h / det_w
        if win_h / win_w > det_ar:
            win_w = int(win_h / det_ar)
        else:
            win_h = int(win_w * det_ar)

        # Don't create windows larger than the standard detector input
        # (those are already handled well by the standard pipeline)
        if win_w >= det_w and win_h >= det_h:
            if verbose:
                _aw_log.info(f"  [OC] obj[{idx}] would yield {win_w}x{win_h} >= det "
                             f"{det_w}x{det_h} → SKIP (covered by standard pipeline)")
            continue

        # Center window on object
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        wx1 = max(0, int(xc - win_w / 2))
        wy1 = max(0, int(yc - win_h / 2))
        wx2 = wx1 + win_w
        wy2 = wy1 + win_h

        # Clamp to image boundaries
        if wx2 > img_w:
            wx2 = img_w
            wx1 = max(0, wx2 - win_w)
        if wy2 > img_h:
            wy2 = img_h
            wy1 = max(0, wy2 - win_h)

        if verbose:
            _aw_log.info(f"  [OC] obj[{idx}] obj=({int(x1)},{int(y1)},{int(x2)},{int(y2)}) "
                         f"→ window=({wx1},{wy1},{wx2},{wy2}) size={win_w}x{win_h}")

        windows.append([wx1, wy1, wx2, wy2])
        sources.append(idx)

    if not windows:
        if return_sources:
            return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.int32)
        return np.empty((0, 4), dtype=np.int32)

    windows = np.array(windows, dtype=np.int32)
    sources = np.array(sources, dtype=np.int32)

    # Deduplicate: remove windows with high IoU (>0.7) with each other
    # Keep the smaller window (tighter crop = better resolution)
    if len(windows) > 1:
        keep = _nms_keep_smaller(windows, iou_threshold=0.7)
        if verbose and len(keep) < len(windows):
            dropped = set(range(len(windows))) - set(keep)
            for d in dropped:
                _aw_log.info(f"  [OC] window from src obj[{int(sources[d])}] "
                             f"SUPPRESSED by NMS (overlap with smaller OC window)")
        windows = windows[keep]
        sources = sources[keep]

    if return_sources:
        return windows, sources
    return windows


def _nms_keep_smaller(windows, iou_threshold=0.7):
    """NMS variant that keeps smaller windows (tighter crops) over larger ones.

    Args:
        windows (np.ndarray): (N, 4) array of windows [x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        list: Indices of windows to keep.
    """
    if len(windows) == 0:
        return []

    areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    # Sort by area ascending (smallest first — we prefer tighter crops).
    # Stable sort: for equal-area windows we keep the input ordering deterministic.
    order = np.argsort(areas, kind='stable')

    keep = []
    suppressed = set()

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)

        # Suppress larger windows that overlap significantly
        for other_idx in order:
            if other_idx in suppressed or other_idx == idx:
                continue

            # Compute IoU
            xx1 = max(windows[idx, 0], windows[other_idx, 0])
            yy1 = max(windows[idx, 1], windows[other_idx, 1])
            xx2 = min(windows[idx, 2], windows[other_idx, 2])
            yy2 = min(windows[idx, 3], windows[other_idx, 3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = areas[idx] + areas[other_idx] - inter

            if union > 0 and inter / union > iou_threshold:
                suppressed.add(other_idx)

    return keep
