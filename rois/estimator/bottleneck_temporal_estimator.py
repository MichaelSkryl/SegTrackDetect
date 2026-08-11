"""
BottleneckTemporalEstimator — uses the rebuilt TemporalUnet (with ConvGRU
at a configurable encoder layer) as a drop-in replacement for the frozen
TorchScript estimator in the SegTrackDetect pipeline.

Supports two weight loading modes:
  - GRU-only weights (Phase 1): loads UNet from TorchScript, GRU from .pt
  - Full model weights (Phase 2): loads entire TemporalUnet from .pt,
    skipping TorchScript loading since all weights are fine-tuned.
"""

import os
import torch

from .configs import ESTIMATOR_MODELS
from .unet_resnet18 import TemporalUnet


class BottleneckTemporalEstimator:
    """
    ROI Estimator that uses the TemporalUnet (UNet-ResNet18 with ConvGRU)
    instead of the frozen TorchScript model.

    API-compatible with the original Estimator class so it can be used as
    a drop-in replacement in ROIModule/fusion.py.

    Args:
        model_name (str): Name of the ROI model (e.g., 'SDS_large').
        device (str): Device to use ('cuda' or 'cpu').
        gru_hidden (int): Hidden channels for the ConvGRU.
        gru_kernel (int): Kernel size for ConvGRU convolutions.
        gru_weights (str or None): Path to trained weights. Can be either:
            - GRU-only weights (from Phase 1 training)
            - Full model weights (from Phase 2 end-to-end training)
            The loader auto-detects which format it is.
        insertion_point (int): Which encoder feature to apply GRU to.
            2 = layer1 (64ch), 3 = layer2 (128ch), 4 = layer3 (256ch),
            5 = layer4/bottleneck (512ch, default).
    """

    def __init__(self, model_name, device='cuda', gru_hidden=64, gru_kernel=3,
                 gru_weights=None, insertion_point=5):
        assert model_name in ESTIMATOR_MODELS.keys(), \
            f'{model_name} not in ESTIMATOR_MODELS.keys()'

        self.config = ESTIMATOR_MODELS[model_name]
        self.device = device

        # --- Auto-detect perturbation type from the saved checkpoint ---
        perturbation_type = 'gru'  # default for backward compatibility
        peek_state = None
        if gru_weights is not None and os.path.isfile(gru_weights):
            peek_state = torch.load(gru_weights, map_location='cpu')
            if isinstance(peek_state, dict):
                keys = list(peek_state.keys())
                if any('bottleneck_gru.input_proj' in k for k in keys):
                    perturbation_type = 'gru'
                elif any('bottleneck_gru.gru.input_proj' in k for k in keys):
                    perturbation_type = 'frozen_gru'
                else:
                    # Only bottleneck_gru.alpha exists → gaussian / dropout / identity.
                    # All three are identity at inference, so 'identity' works for all.
                    perturbation_type = 'identity'
                print(f"Auto-detected perturbation type: {perturbation_type}")
        
        # Build the full model with configurable insertion point
        self.net = TemporalUnet(
            gru_hidden=gru_hidden,
            gru_kernel=gru_kernel,
            insertion_point=insertion_point,
            perturbation_type=perturbation_type,)
        
        # --- Existing weight-loading logic, but with strict=False for safety ---
        is_full_model = False
        if peek_state is not None and isinstance(peek_state, dict):
            has_encoder = any(k.startswith('encoder.') for k in peek_state.keys())
            has_decoder = any(k.startswith('decoder.') for k in peek_state.keys())
            is_full_model = has_encoder and has_decoder

        if is_full_model:
            print(f"Loading full model weights (Phase 2): "
                  f"{os.path.basename(gru_weights)}")
            missing, unexpected = self.net.load_state_dict(peek_state, strict=False)
            if missing:
                print(f"  Missing keys (OK if expected for this perturbation type): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            # Phase 1 or no GRU weights: load UNet from TorchScript first
            torchscript_path = self.config['weights']
            self.net.load_from_torchscript(torchscript_path)

            # Then load GRU weights if provided
            if gru_weights is not None and os.path.isfile(gru_weights):
                print(f"Loading ConvGRU weights (Phase 1): "
                      f"{os.path.basename(gru_weights)}")

                if any('bottleneck_gru' in k for k in state.keys()):
                    # Full model state_dict — extract only GRU params
                    gru_state = {k: v for k, v in state.items()
                                 if 'bottleneck_gru' in k}
                    self.net.load_state_dict(gru_state, strict=False)
                else:
                    # GRU-only state_dict
                    self.net.bottleneck_gru.load_state_dict(state)

        self.net.to(device)
        if perturbation_type == 'gru':
            print(f"  GRU output_proj.weight max: {self.net.bottleneck_gru.output_proj.weight.abs().max().item():.6f}")
            print(f"  GRU alpha gate: {torch.sigmoid(self.net.bottleneck_gru.alpha).item():.6f}")
        self.net.eval()

        # Detect dtype from encoder parameters
        dtypes = {p.dtype for p in self.net.encoder.parameters()}
        self.dtype = next(iter(dtypes))

        # Config accessors (same interface as Estimator)
        self.input_size = self.config['in_size']
        self.preprocess = self.config['preprocess']
        self.preprocess_args = self.config['preprocess_args']
        self.postprocess = self.config['postprocess']

        # Our TemporalUnet outputs raw logits (no sigmoid).
        # Override postprocess_args so unet_postprocess applies sigmoid.
        self.postprocess_args = dict(self.config['postprocess_args'])
        self.postprocess_args['sigmoid_included'] = False

        total_params = sum(p.numel() for p in self.net.parameters())
        gru_params = sum(p.numel() for p in self.net.bottleneck_gru.parameters())
        print(f"BottleneckTemporalEstimator ready: "
              f"total={total_params:,}, GRU={gru_params:,}, "
              f"hidden={gru_hidden}, kernel={gru_kernel}, "
              f"insertion_point={insertion_point}, "
              f"mode={'full_model' if is_full_model else 'gru_only'}")

    def reset_temporal_state(self):
        """Reset the ConvGRU hidden state. Call at each new sequence."""
        self.net.reset_temporal_state()

    @torch.inference_mode()
    def get_estimated_roi(self, img_tensor, orig_shape):
        """
        Estimate the ROI mask with temporal refinement.

        Args:
            img_tensor (torch.Tensor): Preprocessed input (1, 3, H, W).
            orig_shape (tuple): Original image dimensions (H, W).

        Returns:
            torch.Tensor: Binary ROI mask of shape (H_low, W_low).
        """
        output = self.net(img_tensor.to(self.device).to(self.dtype))

        estimated_mask = self.postprocess(
            output,
            orig_shape,
            **self.postprocess_args,
        )
        return estimated_mask
