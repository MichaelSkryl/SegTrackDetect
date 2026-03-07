"""
Training script for temporal ConvGRU — v4 with all critical fixes.

Key fixes over v3:
  1. No sigmoid in TemporalUnet SegmentationHead — output is raw logits.
     Uses BCEWithLogitsLoss for numerically stable gradients.
  2. Truncated BPTT: backprop per-frame, detach hidden state after each step.
     Prevents memory explosion and gradient vanishing over long sequences.
  3. GT supervision: train against ground-truth binary masks (from COCO bbox
     annotations), with self-distillation as regularizer.
     This lets the GRU IMPROVE on the teacher, not just copy it.
  4. Alpha initialized to -5.0 (sigmoid ≈ 0.007) — near-identity start.
  5. Proper insertion_point defaults: 2 for SDS_tiny, 5 for SDS_large.

Loss = 0.7 * BCE(logits, gt_mask) + 0.2 * BCE(logits, teacher) + 0.1 * TC

Usage:

    # ===== Phase 1: GRU-only training =====

    # SDS_large: bottleneck at layer4 (512ch, 14×24 — fine)
    python train_bottleneck_temporal_v4.py \\
        --data_root /SegTrackDetect/data/SeaDronesSee \\
        --roi_model SDS_large --mode bottleneck \\
        --insertion_point 5 --gru_hidden 64 \\
        --epochs 30 --lr 1e-3 --seq_len 16 \\
        --out_dir weights/temporal_large_v4 --phase 1

    # SDS_tiny: bottleneck at layer1 (64ch, 16×24 — good spatial res)
    python train_bottleneck_temporal_v4.py \\
        --data_root /SegTrackDetect/data/SeaDronesSee \\
        --roi_model SDS_tiny --mode bottleneck \\
        --insertion_point 2 --gru_hidden 32 \\
        --epochs 30 --lr 1e-3 --seq_len 16 \\
        --out_dir weights/temporal_tiny_v4 --phase 1

    # SDS_tiny: post_unet approach
    python train_bottleneck_temporal_v4.py \\
        --data_root /SegTrackDetect/data/SeaDronesSee \\
        --roi_model SDS_tiny --mode post_unet --temporal_hidden 16 \\
        --epochs 30 --lr 1e-3 --seq_len 16 \\
        --out_dir weights/temporal_tiny_postunet_v4 --phase 1

    # ===== Phase 2: End-to-end fine-tuning =====

    python train_bottleneck_temporal_v4.py \\
        --data_root /SegTrackDetect/data/SeaDronesSee \\
        --roi_model SDS_large --mode bottleneck \\
        --insertion_point 5 --gru_hidden 64 \\
        --epochs 15 --lr 1e-4 --seq_len 16 \\
        --out_dir weights/temporal_large_e2e_v4 --phase 2 \\
        --gru_weights weights/temporal_large_v4/bottleneck_gru_best.pt \\
        --unet_lr_scale 0.01
"""

import argparse
import os
import json

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from datasets import ROIDataset, DirectoryDataset
from rois.estimator.configs import ESTIMATOR_MODELS
from rois.estimator.unet_resnet18 import TemporalUnet
from rois.estimator.conv_gru import TemporalROIRefiner


def temporal_consistency_loss(pred, prev_pred, weight=0.1):
    """
    Penalize large frame-to-frame changes in the output mask.
    Encourages the ConvGRU to produce stable, smooth masks across time.
    """
    if prev_pred is None:
        return 0.0
    return weight * nn.functional.mse_loss(pred, prev_pred)


def load_gt_mask(metadata, in_size, annotations, device):
    """
    Load ground-truth binary mask from COCO annotations.

    The ROI estimation network is trained as binary segmentation where
    bounding box rectangles are painted as white (1) on black (0) background.

    Args:
        metadata: Batch metadata from dataloader (contains image_id).
        in_size: (H, W) of the ROI estimation network input.
        annotations: Dict mapping image_id → list of bbox [x, y, w, h].
        device: torch device.

    Returns:
        torch.Tensor: Binary mask of shape (1, 1, H, W).
    """
    H, W = in_size
    mask = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)

    img_id = int(metadata['coco']['id'].item())
    if img_id in annotations:
        for bbox in annotations[img_id]:
            x, y, w, h = bbox
            # Scale bbox from original image coords to low-res mask coords
            orig_h = int(metadata['coco']['height'].item())
            orig_w = int(metadata['coco']['width'].item())
            scale_x = W / orig_w
            scale_y = H / orig_h

            x1 = max(0, int(x * scale_x))
            y1 = max(0, int(y * scale_y))
            x2 = min(W, int((x + w) * scale_x))
            y2 = min(H, int((y + h) * scale_y))

            if x2 > x1 and y2 > y1:
                mask[0, 0, y1:y2, x1:x2] = 1.0

    return mask


def build_annotation_index(ds):
    """
    Build a mapping from image_id → list of bounding boxes.

    Args:
        ds: DirectoryDataset instance.

    Returns:
        dict: {image_id: [[x, y, w, h], ...]}
    """
    annos = {}
    for a in ds.annotations.get('annotations', []):
        img_id = a['image_id']
        if img_id not in annos:
            annos[img_id] = []
        annos[img_id].append(a['bbox'])
    return annos


def train():
    parser = argparse.ArgumentParser(
        description='Train temporal ConvGRU v4 (fixed: logits, truncated BPTT, GT supervision)')

    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--roi_model', type=str, default='SDS_tiny')

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Consecutive frames per training subsequence.')
    parser.add_argument('--out_dir', type=str, default='weights/temporal_v4')
    parser.add_argument('--cpu', default=False, action='store_true')

    # Mode selection
    parser.add_argument('--mode', type=str, default='bottleneck',
                        choices=['bottleneck', 'post_unet'],
                        help='bottleneck: GRU inside UNet encoder. '
                             'post_unet: GRU after frozen UNet output.')

    # Insertion point for bottleneck mode
    parser.add_argument('--insertion_point', type=int, default=5,
                        choices=[2, 3, 4, 5],
                        help='For bottleneck mode: which encoder layer. '
                             '2=layer1(64ch,H/4), 3=layer2(128ch,H/8), '
                             '4=layer3(256ch,H/16), 5=layer4(512ch,H/32). '
                             'Use 2 for SDS_tiny, 5 for SDS_large.')

    # GRU params (bottleneck mode)
    parser.add_argument('--gru_hidden', type=int, default=64)
    parser.add_argument('--gru_kernel', type=int, default=3)

    # Post-UNet refiner params
    parser.add_argument('--temporal_hidden', type=int, default=16,
                        help='Hidden channels for post-UNet TemporalROIRefiner.')
    parser.add_argument('--temporal_ks', type=int, default=3,
                        help='Kernel size for post-UNet TemporalROIRefiner.')

    # Phase control
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Phase 1: GRU only. Phase 2: end-to-end.')
    parser.add_argument('--gru_weights', type=str, default=None,
                        help='Path to pretrained GRU/refiner weights.')
    parser.add_argument('--unet_lr_scale', type=float, default=0.01,
                        help='LR multiplier for pretrained UNet in Phase 2.')

    # Loss weights
    parser.add_argument('--gt_weight', type=float, default=0.7,
                        help='Weight for GT supervision loss.')
    parser.add_argument('--distill_weight', type=float, default=0.2,
                        help='Weight for self-distillation loss.')
    parser.add_argument('--tc_weight', type=float, default=0.1,
                        help='Weight for temporal consistency loss.')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = (torch.device('cuda:0')
              if torch.cuda.device_count() > 0 and not args.cpu
              else torch.device('cpu'))

    config = ESTIMATOR_MODELS[args.roi_model]

    # ==================================================================
    # Build model based on mode
    # ==================================================================

    if args.mode == 'post_unet':
        # ----------------------------------------------------------
        # Post-UNet mode: frozen TorchScript UNet + trainable refiner
        # ----------------------------------------------------------
        frozen_unet = torch.jit.load(config['weights'], map_location='cpu')
        frozen_unet.to(device)
        frozen_unet.eval()
        for p in frozen_unet.parameters():
            p.requires_grad = False

        # Detect dtype
        unet_dtype = next(iter({p.dtype for p in frozen_unet.parameters()}))

        refiner = TemporalROIRefiner(
            hidden_channels=args.temporal_hidden,
            kernel_size=args.temporal_ks,
        ).to(device)

        if args.gru_weights and os.path.isfile(args.gru_weights):
            print(f"Loading pretrained refiner weights: {args.gru_weights}")
            refiner.load_state_dict(
                torch.load(args.gru_weights, map_location='cpu'))

        trainable_params = list(refiner.parameters())
        optimizer = optim.Adam(trainable_params, lr=args.lr)

    else:
        # ----------------------------------------------------------
        # Bottleneck mode — use TemporalUnet (NO sigmoid in head)
        # ----------------------------------------------------------
        model = TemporalUnet(
            gru_hidden=args.gru_hidden,
            gru_kernel=args.gru_kernel,
            insertion_point=args.insertion_point,
        )
        model.load_from_torchscript(config['weights'])

        if args.gru_weights and os.path.isfile(args.gru_weights):
            print(f"Loading pretrained GRU weights: {args.gru_weights}")
            gru_state = torch.load(args.gru_weights, map_location='cpu')
            model.bottleneck_gru.load_state_dict(gru_state)

        if args.phase == 1:
            model.freeze_unet()
            trainable_params = [p for p in model.parameters()
                                if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=args.lr)
        else:
            param_groups = model.unfreeze_unet(lr_scale=args.unet_lr_scale)
            optimizer = optim.Adam([
                {'params': param_groups[0]['params'], 'lr': args.lr},
                {'params': param_groups[1]['params'],
                 'lr': args.lr * args.unet_lr_scale},
            ])
            trainable_params = [p for p in model.parameters()
                                if p.requires_grad]

        model.to(device)

    # ==================================================================
    # Preprocessing & dataset
    # ==================================================================
    from rois.estimator.configs.common import estimator_preprocess
    roi_transform = estimator_preprocess(**config['preprocess_args'])

    ds = DirectoryDataset(data_root=args.data_root, split=args.split)

    # Build annotation index for GT supervision
    anno_index = build_annotation_index(ds)
    print(f"  Annotation index: {len(anno_index)} images with annotations")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # BCEWithLogitsLoss for numerically stable training on raw logits
    bce_loss = nn.BCEWithLogitsLoss()

    # ==================================================================
    # Print summary
    # ==================================================================
    print(f"\nTraining temporal module — Mode: {args.mode}, Phase {args.phase}")
    print(f"  Model: {args.roi_model} (input: {config['in_size']})")

    if args.mode == 'post_unet':
        print(f"  Post-UNet refiner: hidden={args.temporal_hidden}, "
              f"kernel={args.temporal_ks}")
    else:
        layer_name = f"layer{args.insertion_point - 1}" if args.insertion_point <= 4 else 'layer4 (bottleneck)'
        ch = TemporalUnet.CHANNEL_MAP.get(args.insertion_point)
        h, w = config['in_size']
        divisor = {2: 4, 3: 8, 4: 16, 5: 32}[args.insertion_point]
        spatial = f"{h // divisor}×{w // divisor}"
        print(f"  GRU insertion: {layer_name} ({ch}ch, {spatial})")
        print(f"  GRU hidden: {args.gru_hidden}, kernel: {args.gru_kernel}")

    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Sequences: {len(ds.seq2images)}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Loss: {args.gt_weight}*BCE(logits,GT) + "
          f"{args.distill_weight}*BCE(logits,teacher) + "
          f"{args.tc_weight}*TC")
    if args.phase == 2:
        print(f"  UNet LR scale: {args.unet_lr_scale}")
    print()

    # ==================================================================
    # Training loop — truncated BPTT (per-frame backprop)
    # ==================================================================
    best_loss = float('inf')

    for epoch in range(args.epochs):

        # --- Set training modes ---
        if args.mode == 'post_unet':
            refiner.train()
        else:
            model.train()
            if args.phase == 1:
                model.encoder.eval()
                model.decoder.eval()
                model.segmentation_head.eval()
                model.bottleneck_gru.train()
            else:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

        epoch_losses = []

        for seq_name, seq_flist in tqdm(ds.seq2images.items(),
                                         desc=f"Epoch {epoch+1}/{args.epochs}"):
            seq_flist = sorted(seq_flist)
            if len(seq_flist) < 2:
                continue

            for start_idx in range(0, len(seq_flist), args.seq_len):
                sub_flist = seq_flist[start_idx:start_idx + args.seq_len]
                if len(sub_flist) < 2:
                    continue

                dataset = ROIDataset(
                    sub_flist, ds, config['in_size'], roi_transform)
                dataloader = DataLoader(
                    dataset, batch_size=1, shuffle=False, num_workers=4)

                # Reset temporal state for each subsequence
                if args.mode == 'post_unet':
                    refiner.reset_hidden_state()
                else:
                    model.reset_temporal_state()

                prev_output = None

                for img, metadata in dataloader:
                    img = img.to(device).float()
                    img_roi = roi_transform(img)

                    # Load GT mask for this frame
                    gt_mask = load_gt_mask(
                        metadata, config['in_size'], anno_index, device)

                    if args.mode == 'post_unet':
                        # Teacher: frozen TorchScript UNet (post-sigmoid)
                        with torch.no_grad():
                            teacher_out = frozen_unet(
                                img_roi.to(unet_dtype))
                            teacher_out = teacher_out.float()

                        # Student: frozen UNet output + trainable refiner
                        student_out = refiner(teacher_out)

                        # For post_unet mode, teacher is post-sigmoid.
                        # The refiner output is a blend, not pure logits.
                        # Use MSE for distillation, BCE for GT.
                        distill_loss = nn.functional.mse_loss(
                            student_out, teacher_out.detach())
                        # For GT, apply sigmoid to convert teacher-scale to prob
                        gt_loss = nn.functional.binary_cross_entropy_with_logits(
                            student_out, gt_mask)

                    else:
                        # Teacher: our UNet WITHOUT GRU (raw logits, no sigmoid)
                        with torch.no_grad():
                            teacher_out = model.forward_without_gru(img_roi)

                        # Student: our UNet WITH GRU (raw logits)
                        student_out = model(img_roi)

                        # Both are logits — BCE with logits for both
                        # Teacher target: apply sigmoid to convert logits → prob
                        teacher_prob = torch.sigmoid(teacher_out.detach())
                        gt_loss = bce_loss(student_out, gt_mask)
                        distill_loss = bce_loss(student_out, teacher_prob)

                    # Temporal consistency on the output
                    tc_loss = temporal_consistency_loss(
                        student_out, prev_output, weight=1.0)

                    # Combined loss
                    loss = (args.gt_weight * gt_loss
                            + args.distill_weight * distill_loss
                            + args.tc_weight * tc_loss)

                    # === Truncated BPTT: backprop each frame independently ===
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=1.0)
                    optimizer.step()

                    # Detach hidden state to prevent graph accumulation
                    if args.mode == 'post_unet':
                        refiner.detach_hidden_state()
                    else:
                        model.detach_temporal_state()

                    prev_output = student_out.detach()
                    epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        lr_str = ", ".join(
            [f"{pg['lr']:.6f}" for pg in optimizer.param_groups])
        print(f"Epoch {epoch+1}/{args.epochs} — loss: {mean_loss:.6f}, "
              f"lr: [{lr_str}]")
        if args.mode == 'bottleneck':
            alpha_val = torch.sigmoid(model.bottleneck_gru.alpha).item()
            print(f"  Alpha gate: {alpha_val:.4f} (sigmoid of {model.bottleneck_gru.alpha.item():.3f})")

        if mean_loss < best_loss:
            best_loss = mean_loss
            if args.mode == 'post_unet':
                save_path = os.path.join(
                    args.out_dir, 'temporal_refiner_best.pt')
                torch.save(refiner.state_dict(), save_path)
            elif args.phase == 1:
                save_path = os.path.join(
                    args.out_dir, 'bottleneck_gru_best.pt')
                torch.save(model.bottleneck_gru.state_dict(), save_path)
            else:
                save_path = os.path.join(
                    args.out_dir, 'full_model_best.pt')
                torch.save(model.state_dict(), save_path)
            print(f"  → Saved best: {save_path} (loss: {best_loss:.6f})")

    # Save final
    if args.mode == 'post_unet':
        save_path = os.path.join(args.out_dir, 'temporal_refiner_final.pt')
        torch.save(refiner.state_dict(), save_path)
    elif args.phase == 1:
        save_path = os.path.join(args.out_dir, 'bottleneck_gru_final.pt')
        torch.save(model.bottleneck_gru.state_dict(), save_path)
    else:
        save_path = os.path.join(args.out_dir, 'full_model_final.pt')
        torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Final model: {save_path}")


if __name__ == '__main__':
    train()
