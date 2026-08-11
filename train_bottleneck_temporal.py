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
import torchvision.models as models


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
    parser.add_argument('--bptt_steps', type=int, default=4,
                        help='Number of frames to backpropagate through (Truncated BPTT).')
    parser.add_argument('--out_dir', type=str, default='weights/temporal_v4')
    parser.add_argument('--cpu', default=False, action='store_true')

    # Mode selection
    parser.add_argument('--mode', type=str, default='bottleneck',
                        choices=['bottleneck', 'post_unet', 'baseline'],
                        help='bottleneck: GRU inside UNet encoder. '
                             'post_unet: GRU after frozen UNet output.'
                             'baseline: Train purely spatial UNet from scratch.')

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
                        
    # Perturbations
    parser.add_argument('--perturbation_type', type=str, default='gru', choices=['gru', 'gaussian', 'dropout', 'frozen_gru', 'identity'], help='Layer-1 perturbation source.')
    parser.add_argument('--perturb_std', type=float, default=0.05, help='Gaussian std (gaussian only).')
    parser.add_argument('--perturb_p', type=float, default=0.05, help='Dropout2d probability (dropout only).')
    parser.add_argument('--perturb_alpha_init', type=float, default=-5.0, help='Frozen GRU alpha init (frozen_gru only).')
    
    parser.add_argument('--bn_eval', default=False, action='store_true', help='Force BN to eval mode during baseline training (BN-mode ablation).')
    parser.add_argument('--baseline_batch_size', type=int, default=8, help='Batch size for baseline training (BN-friendly).')
    parser.add_argument('--alpha_init', type=float, default=-3.0, help='Alpha initialization (Controls the GRU output to the original features')
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name (looks for data_root/val_split.json).')
    parser.add_argument('--val_every', type=int, default=1, help='Run validation every N epochs.')
    parser.add_argument('--select_by', type=str, default='val_loss', choices=['train_loss', 'val_loss'], help='Metric for best-checkpoint selection.')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


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
        # Bottleneck OR Baseline mode — use TemporalUnet
        # ----------------------------------------------------------
        # If baseline, gru_hidden/kernel don't matter, but we initialize anyway
        pkw = {}
        if args.perturbation_type == 'gaussian':
            pkw = {'std': args.perturb_std}
        elif args.perturbation_type == 'dropout':
            pkw = {'p': args.perturb_p}
        elif args.perturbation_type == 'frozen_gru':
            pkw = {'alpha_init': args.perturb_alpha_init}
        
        model = TemporalUnet(gru_hidden=args.gru_hidden, gru_kernel=args.gru_kernel, insertion_point=args.insertion_point, 
            perturbation_type=args.perturbation_type, perturbation_kwargs=pkw, alpha_init=args.alpha_init)
        weight_path = config.get('weights', '')

        # --- Safe Weight Loading & ImageNet Injection ---
        if os.path.exists(weight_path):
            print(f"Loading UNet baseline weights from {weight_path}")
            state = torch.load(weight_path, map_location='cpu')

            # Check if it's a standard PyTorch state_dict (from our baseline script)
            if isinstance(state, dict) and 'encoder.conv1.weight' in state:
                filtered_state = {k: v for k, v in state.items() if 'bottleneck_gru' not in k}
                model.load_state_dict(filtered_state, strict=False)
                
             #   if args.phase == 1 and hasattr(model, 'bottleneck_gru'):
              #      print("Applying Near-Zero Initialization to ConvGRU...")
              #      torch.nn.init.normal_(model.bottleneck_gru.output_proj.weight, mean=0.0, std=1e-4)
              #      torch.nn.init.zeros_(model.bottleneck_gru.output_proj.bias)
            else:
                # Fallback to original TorchScript format
                model.load_from_torchscript(weight_path)
        else:
            if args.mode == 'baseline':
                print(f"No weights found at '{weight_path}'.")
                print("Injecting ImageNet pre-trained weights into the UNet Encoder...")
                # Download/load ImageNet weights
                resnet18_pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                # Inject into our custom encoder (strict=False ignores FC layer)
                model.encoder.load_state_dict(resnet18_pretrained.state_dict(), strict=False)
                print("ImageNet weights successfully loaded into the backbone! Decoder remains random.")
            else:
                raise FileNotFoundError(
                    f"Cannot run Phase 1 or 2 without baseline weights at '{weight_path}'! Please run '--mode baseline' first.")

        if args.gru_weights and os.path.isfile(args.gru_weights) and args.perturbation_type == 'gru':
            print(f"Loading pretrained GRU weights: {args.gru_weights}")
            gru_state = torch.load(args.gru_weights, map_location='cpu')
            model.bottleneck_gru.load_state_dict(gru_state)
        
        # Force alpha to the CLI-requested value, even if a saved checkpoint had a
        # different alpha stored. CLI must win over loaded weights.
        if (args.perturbation_type == 'gru' and not (args.gru_weights and os.path.isfile(args.gru_weights)) 
            and hasattr(model, 'bottleneck_gru') 
            and hasattr(model.bottleneck_gru, 'alpha')):
            model.bottleneck_gru.alpha.data = torch.tensor(float(args.alpha_init), device=device)
            print(f"  Override: GRU alpha set to {args.alpha_init} "
                  f"(sigmoid = {torch.sigmoid(model.bottleneck_gru.alpha).item():.4f})")
        
        if args.mode == 'baseline':
            # Train all spatial UNet layers, ignore GRU
            trainable_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(
                model.segmentation_head.parameters())
            optimizer = optim.Adam(trainable_params, lr=args.lr)
        else:
            if args.phase == 1:
                model.freeze_unet()
                if hasattr(model, 'bottleneck_gru'):
                    if args.perturbation_type == 'gru':
                    # Zero-init output_proj so temporal_out ≈ 0 at start
                        torch.nn.init.zeros_(model.bottleneck_gru.output_proj.weight)
                        torch.nn.init.zeros_(model.bottleneck_gru.output_proj.bias)
                    # Drive alpha very negative so gate ≈ 0 → refined ≈ x exactly
                        model.bottleneck_gru.alpha.data = torch.tensor(args.alpha_init, device=device)
                        print(f"ConvGRU initialized as near-identity (output_proj=0, alpha={args.alpha_init})")
                trainable_params = [p for p in model.parameters() if p.requires_grad]
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
                print(f'Length of trainable params: {len(trainable_params)}')

        model.to(device)

    # ==================================================================
    # Preprocessing & dataset
    # ==================================================================
    from rois.estimator.configs.common import estimator_preprocess
    roi_transform = estimator_preprocess(**config['preprocess_args'])

    ds = DirectoryDataset(data_root=args.data_root, split=args.split)
    
    val_ds = None
    val_anno_index = None
    if args.select_by == 'val_loss':
        try:
            val_ds = DirectoryDataset(data_root=args.data_root, split=args.val_split)
            val_anno_index = build_annotation_index(val_ds)
            print(f"  Val dataset loaded: {len(val_ds.images)} images, "
                  f"{len(val_anno_index)} with annotations")
        except SystemExit:
            print(f"  WARNING: val split '{args.val_split}' not found. "
                  f"Falling back to train_loss selection.")
            args.select_by = 'train_loss'
            val_ds = None

    # Build annotation index for GT supervision
    anno_index = build_annotation_index(ds)
    print(f"  Annotation index: {len(anno_index)} images with annotations")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # BCEWithLogitsLoss for numerically stable training on raw logits
    bce_loss = nn.BCEWithLogitsLoss()
    
    model.eval()  # Full eval — no train-mode quirks
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for seq_name, seq_flist in ds.seq2images.items():
            seq_flist = sorted(seq_flist)
            dataset = ROIDataset(seq_flist[:16], ds, config['in_size'], roi_transform)
            for img, metadata in DataLoader(dataset, batch_size=1, shuffle=False):
                img = img.to(device).float()
                img_roi = roi_transform(img)
                out = model.forward_without_gru(img_roi)
                gm = load_gt_mask(metadata, config['in_size'], anno_index, device)
                total_loss += bce_loss(out, gm).item()
                n += 1
                print(f"  teacher out: min={out.min().item():.2f} max={out.max().item():.2f} "
                      f"mean={out.mean().item():.2f} | gt_loss={total_loss / n:.4f}")
                if n >= 5:
                    break
            if n >= 5:
                break
    print(f"Pre-train mean teacher loss on train data: {total_loss / n:.6f}")

    # ==================================================================
    # Print summary
    # ==================================================================
    print(f"\nTraining temporal module — Mode: {args.mode}, Phase {args.phase}")
    print(f"  Model: {args.roi_model} (input: {config['in_size']})")

    if args.mode == 'post_unet':
        print(f"  Post-UNet refiner: hidden={args.temporal_hidden}, kernel={args.temporal_ks}")
    elif args.mode == 'baseline':
        print("  Architecture: Pure Spatial UNet (No GRU)")
    else:
        # Only print GRU details for phase 1 and 2
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
    
    if args.mode == 'baseline':
        print("  Loss: BCE(logits, GT) [Pure Spatial Baseline]")
    else:
        print(f"  Loss: {args.gt_weight}*BCE(logits,GT) + {args.distill_weight}*BCE(logits,teacher) + {args.tc_weight}*TC")
        if args.phase == 2:
            print(f"  UNet LR scale: {args.unet_lr_scale}")
    print()

    # ==================================================================
    # Training loop — truncated BPTT (per-frame backprop)
    # ==================================================================
    best_loss = float('inf')
    debug_printed = False

    if args.mode == 'baseline':
        # Build one big dataset over all frames
        all_flist = []
        for seq_name, seq_flist in ds.seq2images.items():
            all_flist.extend(sorted(seq_flist))
        flat_dataset = ROIDataset(all_flist, ds, config['in_size'], roi_transform)
        flat_loader = DataLoader(
            flat_dataset, batch_size=args.baseline_batch_size,
            shuffle=True, num_workers=4, drop_last=True)

    for epoch in range(args.epochs):

        # --- Set training modes ---
        if args.mode == 'post_unet':
            refiner.train()
        else:
            model.train()
            if args.mode == 'baseline':
                # Train mode for everyhing — let BN learn running stats
                if args.bn_eval:
                    for module in model.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            module.eval()
            elif args.phase == 1:
                model.encoder.eval()
                model.decoder.eval()
                model.segmentation_head.eval()
                model.bottleneck_gru.train()
            else:
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

        epoch_losses = []
        if args.mode == 'baseline':
            for img, metadata in tqdm(flat_loader,
                                      desc=f"Epoch {epoch + 1}/{args.epochs}"):
                img = img.to(device).float()
                img_roi = roi_transform(img)

                # Build batched gt_masks by iterating over the batch dimension
                B = img.shape[0]
                gt_masks = []
                for i in range(B):
                    sample_meta = {'coco': {k: metadata['coco'][k][i:i + 1]
                                            for k in metadata['coco']}}
                    gt_masks.append(
                        load_gt_mask(sample_meta, config['in_size'],
                                     anno_index, device).squeeze(0)
                    )
                gt_masks = torch.stack(gt_masks)  # shape (B, 1, H, W)

                student_out = model.forward_without_gru(img_roi)
                loss = bce_loss(student_out, gt_masks)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optimizer.step()
                epoch_losses.append(loss.item())
        else:
            for seq_name, seq_flist in tqdm(ds.seq2images.items(),
                                            desc=f"Epoch {epoch + 1}/{args.epochs}"):
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

                    optimizer.zero_grad()
                    accumulated_loss = 0
                    chunk_frames = 0

                    for step_idx, (img, metadata) in enumerate(dataloader):
                        img = img.to(device).float()
                        img_roi = roi_transform(img)

                        # Load GT mask for this frame
                        gt_mask = load_gt_mask(
                            metadata, config['in_size'], anno_index, device)

                        if not debug_printed:
                            print("\n===== DEBUG =====")
                            print("gt_mask.sum():")
                            print(gt_mask.sum().item())

                            if hasattr(model, 'bottleneck_gru') and args.perturbation_type == 'gru':
                                print("\noutput_proj.weight.abs().max():")
                                print(model.bottleneck_gru.output_proj.weight.abs().max().item())

                            print("=================\n")
                            debug_printed = True

                        if args.mode == 'post_unet':
                            # Teacher: frozen TorchScript UNet (post-sigmoid)
                            with torch.no_grad():
                                teacher_out = frozen_unet(
                                    img_roi.to(unet_dtype))
                                teacher_out = teacher_out.float()

                            # Student: frozen UNet output + trainable refiner
                            student_out = refiner(teacher_out)

                            # For post_unet mode, teacher is post-sigmoid.
                            distill_loss = nn.functional.mse_loss(student_out, teacher_out.detach())
                            gt_loss = nn.functional.binary_cross_entropy_with_logits(student_out, gt_mask)
                            tc_loss = temporal_consistency_loss(student_out, prev_output, weight=1.0)
                        elif args.mode == 'baseline':
                            student_out = model.forward_without_gru(img_roi)
                            gt_loss = bce_loss(student_out, gt_mask)
                            distill_loss = 0.0
                            tc_loss = 0.0
                        else:
                            # Teacher: our UNet WITHOUT GRU (raw logits, no sigmoid)
                            with torch.no_grad():
                                teacher_out = model.forward_without_gru(img_roi)

                            # Student: our UNet WITH GRU (raw logits)
                            student_out = model(img_roi)

                            teacher_prob = torch.sigmoid(teacher_out.detach())
                            gt_loss = bce_loss(student_out, gt_mask)
                            distill_loss = bce_loss(student_out, teacher_prob)
                            tc_loss = temporal_consistency_loss(student_out, prev_output, weight=1.0)

                        if args.mode == 'baseline':
                            loss = gt_loss
                        else:
                            loss = (
                                        args.gt_weight * gt_loss + args.distill_weight * distill_loss + args.tc_weight * tc_loss)

                        accumulated_loss = accumulated_loss + loss
                        chunk_frames += 1

                        # === Truncated BPTT: backprop every K frames ===
                        if (step_idx + 1) % args.bptt_steps == 0 or (step_idx + 1) == len(dataloader):
                            # Average loss over the chunk to keep gradient scale consistent
                            chunk_loss = accumulated_loss / chunk_frames
                            chunk_loss.backward()

                            torch.nn.utils.clip_grad_norm_(
                                trainable_params, max_norm=5.0)
                            optimizer.step()
                            optimizer.zero_grad()

                            # Detach hidden state to prevent graph accumulation across chunks
                            if args.mode == 'post_unet':
                                refiner.detach_hidden_state()
                            else:
                                model.detach_temporal_state()

                            epoch_losses.append(chunk_loss.item())

                            # Reset accumulators for the next chunk
                            accumulated_loss = 0
                            chunk_frames = 0

                        # Keep prev_output detached so TC loss doesn't backprop into the previous frame
                        # if it crosses a chunk boundary
                        prev_output = student_out.detach()

        scheduler.step()
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        lr_str = ", ".join(
            [f"{pg['lr']:.6f}" for pg in optimizer.param_groups])
        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {mean_loss:.6f}, "
              f"lr: [{lr_str}]")
        if args.mode == 'bottleneck':
            alpha_val = torch.sigmoid(model.bottleneck_gru.alpha).item()
            print(f"  Alpha gate: {alpha_val:.4f} (sigmoid of {model.bottleneck_gru.alpha.item():.3f})")

        val_loss = None
        if val_ds is not None and ((epoch + 1) % args.val_every == 0):
            val_loss = evaluate_val_loss(model, val_ds, config, roi_transform,
                                         val_anno_index, bce_loss, device)
            print(f"  Val loss: {val_loss:.6f}")

        # Decide which metric drives "best"
        selection_metric = val_loss if (args.select_by == 'val_loss' and val_loss is not None) else mean_loss
        if selection_metric < best_loss:
            best_loss = selection_metric
            if args.mode == 'post_unet':
                save_path = os.path.join(args.out_dir, 'temporal_refiner_best.pt')
                torch.save(refiner.state_dict(), save_path)
            elif args.phase == 1 and args.mode != 'baseline':
                save_path = os.path.join(args.out_dir, 'bottleneck_gru_best.pt')
                torch.save(model.bottleneck_gru.state_dict(), save_path)
            else:
                save_path = os.path.join(args.out_dir, 'full_model_best.pt')
                torch.save(model.state_dict(), save_path)
            print(f"  → Saved best ({args.select_by}={best_loss:.6f}): {save_path}")
    # Save final
    if args.mode == 'post_unet':
        save_path = os.path.join(args.out_dir, 'temporal_refiner_final.pt')
        torch.save(refiner.state_dict(), save_path)
    elif args.phase == 1 and args.mode != 'baseline':  # <--- AND HERE
        save_path = os.path.join(args.out_dir, 'bottleneck_gru_final.pt')
        torch.save(model.bottleneck_gru.state_dict(), save_path)
    else:
        save_path = os.path.join(args.out_dir, 'full_model_final.pt')
        torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Final model: {save_path}")


if __name__ == '__main__':
    train()
