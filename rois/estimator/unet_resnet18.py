"""
Exact PyTorch reimplementation of the segmentation_models_pytorch (SMP)
Unet with ResNet18 encoder, matching the TorchScript models shipped with
SegTrackDetect.

The architecture was reverse-engineered from the TorchScript state_dict:
  - Encoder: ResNet18 (conv1→bn1→relu→maxpool→layer1-4)
  - Decoder: 5 SMP-style DecoderBlocks (Conv2dReLU + Identity attention)
  - SegmentationHead: Conv2d(16→1, 3×3) — NO Sigmoid (moved to postprocessing)

KEY FIX vs v3:
  The original TorchScript model has Sigmoid baked into the SegmentationHead.
  During training, this causes double-sigmoid and vanishing gradients. By
  removing Sigmoid from our reimplementation and relying on the postprocessing
  pipeline (with sigmoid_included=False), we get proper gradient flow through
  the ConvGRU.

  During inference, the postprocess function handles sigmoid + thresholding.

Weight loading:
  The `load_from_torchscript()` method loads the encoder, decoder, and
  segmentation head weights from the original TorchScript `.pt` file.
  The final Sigmoid layer in TorchScript maps to nn.Identity() here,
  so there's no shape mismatch — Sigmoid has no parameters.
  The ConvGRU is initialized randomly and must be trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_gru import ConvGRUCell


# ============================================================================
# Encoder: ResNet18 (matching SMP's ResNetEncoder)
# ============================================================================

class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2 × 3×3 conv with skip connection)."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """
    ResNet18 encoder that produces feature maps at 5 scales.

    Returns (in forward order):
        features[0]: input image (B, 3, H, W)        — not used by decoder
        features[1]: after conv1+bn1+relu (B, 64, H/2, W/2)
        features[2]: after layer1 (B, 64, H/4, W/4)
        features[3]: after layer2 (B, 128, H/8, W/8)
        features[4]: after layer3 (B, 256, H/16, W/16)
        features[5]: after layer4 (B, 512, H/32, W/32)  ← bottleneck
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = [x]                          # [0] input

        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)                      # [1] (64, H/2, W/2)

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)                      # [2] (64, H/4, W/4)

        x = self.layer2(x)
        features.append(x)                      # [3] (128, H/8, W/8)

        x = self.layer3(x)
        features.append(x)                      # [4] (256, H/16, W/16)

        x = self.layer4(x)
        features.append(x)                      # [5] (512, H/32, W/32)

        return features


# ============================================================================
# Decoder: SMP-style decoder blocks
# ============================================================================

class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU — matches SMP's Conv2dReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    Single SMP-style decoder block.

    Upsamples the input by 2×, concatenates with skip connection, then
    applies two Conv2dReLU blocks.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels)
        self.attention1 = nn.Identity()
        self.conv2 = Conv2dReLU(out_channels, out_channels)
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            # Handle size mismatches (can happen with odd input dimensions)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    """
    SMP-style UNet decoder with 5 blocks.

    Channel flow (derived from state_dict):
        Block 0: (512 + 256) = 768 → 256
        Block 1: (256 + 128) = 384 → 128
        Block 2: (128 + 64)  = 192 → 64
        Block 3: (64 + 64)   = 128 → 32
        Block 4: (32 + 0)    = 32  → 16  (no skip connection)
    """

    def __init__(self):
        super().__init__()
        self.center = nn.Identity()
        self.blocks = nn.ModuleList([
            DecoderBlock(in_channels=512, skip_channels=256, out_channels=256),
            DecoderBlock(in_channels=256, skip_channels=128, out_channels=128),
            DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64),
            DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32),
            DecoderBlock(in_channels=32,  skip_channels=0,   out_channels=16),
        ])

    def forward(self, features):
        """
        Args:
            features: list of encoder features [input, stem, layer1, layer2, layer3, layer4]
                      indices:                  [0,     1,    2,      3,      4,      5    ]
        """
        skips = [
            features[4],  # 256, H/16  → block 0 skip
            features[3],  # 128, H/8   → block 1 skip
            features[2],  # 64, H/4    → block 2 skip
            features[1],  # 64, H/2    → block 3 skip
            None,         #            → block 4 no skip
        ]

        x = self.center(features[5])  # bottleneck: (512, H/32, W/32)

        for i, (block, skip) in enumerate(zip(self.blocks, skips)):
            x = block(x, skip)

        return x


# ============================================================================
# Segmentation Head — NO SIGMOID
# ============================================================================

class SegmentationHead(nn.Sequential):
    """
    Conv2d(16→1, 3×3) — NO sigmoid applied here.

    KEY FIX: The original TorchScript model bakes nn.Sigmoid() into index [2]
    of this Sequential. Our reimplementation uses nn.Identity() instead, so
    the model outputs raw logits. This is critical because:

      1. During training, BCEWithLogitsLoss needs raw logits (not post-sigmoid)
         for numerically stable gradients.
      2. The postprocessing pipeline (unet_postprocess) handles sigmoid when
         called with sigmoid_included=False.
      3. When loading TorchScript weights, nn.Sigmoid() has no parameters,
         so there's no shape mismatch — it just maps to our nn.Identity().
    """

    def __init__(self, in_channels=16, out_channels=1, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Identity(),   # placeholder (matches state_dict index [1])
            nn.Identity(),   # was nn.Sigmoid() — now handled in postprocessing
        )


# ============================================================================
# Bottleneck ConvGRU
# ============================================================================

class BottleneckConvGRU(nn.Module):
    """
    ConvGRU module operating at a configurable point in the UNet encoder.

    Spatial sizes for each layer (SDS_tiny 64×96 input):
        features[2] = layer1: 64ch,  16×24  ← RECOMMENDED for tiny
        features[3] = layer2: 128ch,  8×12
        features[4] = layer3: 256ch,  4×6
        features[5] = layer4: 512ch,  2×3   ← too small!

    Spatial sizes for each layer (SDS_large 448×768 input):
        features[2] = layer1: 64ch,  112×192
        features[3] = layer2: 128ch,  56×96
        features[4] = layer3: 256ch,  28×48
        features[5] = layer4: 512ch,  14×24  ← fine for large

    Args:
        bottleneck_channels (int): Number of channels at the insertion point.
        hidden_channels (int): Number of channels in the GRU hidden state.
        kernel_size (int): Kernel size for GRU convolutions. Default: 3.
    """

    def __init__(self, bottleneck_channels=512, hidden_channels=64, kernel_size=3, alpha_init=-3.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.bottleneck_channels = bottleneck_channels

        # Project bottleneck_channels → hidden (reduce dimensionality for the GRU)
        self.input_proj = nn.Conv2d(bottleneck_channels, hidden_channels,
                                    kernel_size=1, bias=True)

        # ConvGRU cell operating in the hidden space
        self.gru_cell = ConvGRUCell(hidden_channels, hidden_channels, kernel_size)

        # Project hidden → bottleneck_channels (restore dimensionality)
        self.output_proj = nn.Conv2d(hidden_channels, bottleneck_channels,
                                     kernel_size=1, bias=True)

        # Initialize alpha to -5.0 so sigmoid(-5) ≈ 0.007.
        # The model starts as ~99.3% identity (pure pretrained features)
        # and gradually learns to incorporate temporal information.
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self._hidden_state = None

    def reset_hidden_state(self):
        """Reset at the start of each new video sequence."""
        self._hidden_state = None

    def detach_hidden_state(self):
        """Detach hidden state from computation graph for truncated BPTT."""
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    def forward(self, x):
        """
        Args:
            x: Features at the insertion point (B, C, H, W)
        Returns:
            Temporally refined features (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Initialize hidden state if needed
        if (self._hidden_state is None
                or self._hidden_state.shape[0] != B
                or self._hidden_state.shape[-2:] != (H, W)):
            self._hidden_state = torch.zeros(
                B, self.hidden_channels, H, W,
                device=x.device, dtype=x.dtype,
            )

        # Project to hidden space
        h_in = self.input_proj(x)  # (B, hidden, H, W)

        # GRU step with temporal memory
        self._hidden_state = self.gru_cell(h_in, self._hidden_state)

        # Project back to original channel count
        temporal_out = self.output_proj(self._hidden_state)  # (B, C, H, W)

        # Residual blend: sigmoid(alpha) controls temporal contribution
        # alpha starts at -5.0 → sigmoid(-5) ≈ 0.007, so ~99.3% pretrained
        gate = torch.sigmoid(self.alpha)
        refined = (1 - gate) * x + gate * temporal_out

        return refined


# ============================================================================
# Complete Model: UNet + ConvGRU at configurable insertion point
# ============================================================================

class TemporalUnet(nn.Module):
    """
    Full UNet-ResNet18 with a ConvGRU at a configurable encoder layer for
    temporal ROI estimation.

    Data flow:
        Input (B,3,H,W)
          → Encoder (ResNet18) → features at 5 scales
          → ConvGRU refines features[insertion_point]
          → Decoder (5 blocks with skip connections)
          → SegmentationHead → raw logits (B, 1, H, W)   [NO sigmoid]

    The postprocessing pipeline applies sigmoid + threshold + dilation.

    Args:
        gru_hidden (int): Hidden channels for the ConvGRU. Default: 64.
        gru_kernel (int): Kernel size for GRU convolutions. Default: 3.
        insertion_point (int): Which encoder feature to apply GRU to.
            2 = layer1 output (64ch, H/4×W/4)   ← recommended for SDS_tiny
            3 = layer2 output (128ch, H/8×W/8)
            4 = layer3 output (256ch, H/16×W/16)
            5 = layer4 output (512ch, H/32×W/32) ← default (for SDS_large)
    """

    # Maps insertion_point index → channel count at that encoder feature
    CHANNEL_MAP = {2: 64, 3: 128, 4: 256, 5: 512}

    def __init__(self, gru_hidden=64, gru_kernel=3, insertion_point=5,
                 perturbation_type='gru', perturbation_kwargs=None, alpha_init=-3.0):
        """
        Args:
            perturbation_type: one of 'gru', 'gaussian', 'dropout',
                'frozen_gru', 'identity'.
            perturbation_kwargs: dict passed to the perturbation module.
        """
        super().__init__()
        assert insertion_point in self.CHANNEL_MAP
        self.encoder = ResNetEncoder()
        self.insertion_point = insertion_point
        self.perturbation_type = perturbation_type

        bottleneck_ch = self.CHANNEL_MAP[insertion_point]
        kw = dict(perturbation_kwargs or {})

        # Lazy import to avoid circular dep
        from .perturbation_modules import (
            IdentityPerturbation, GaussianPerturbation,
            Dropout2dPerturbation, FrozenRandomGRUPerturbation,)

        if perturbation_type == 'gru':
            self.bottleneck_gru = BottleneckConvGRU(
                bottleneck_channels=bottleneck_ch,
                hidden_channels=gru_hidden,
                kernel_size=gru_kernel, alpha_init=alpha_init,)
                
        elif perturbation_type == 'gaussian':
            self.bottleneck_gru = GaussianPerturbation(**kw)
        elif perturbation_type == 'dropout':
            self.bottleneck_gru = Dropout2dPerturbation(**kw)
        elif perturbation_type == 'frozen_gru':
            kw.setdefault('bottleneck_channels', bottleneck_ch)
            kw.setdefault('hidden_channels', gru_hidden)
            kw.setdefault('kernel_size', gru_kernel)
            self.bottleneck_gru = FrozenRandomGRUPerturbation(**kw)
        elif perturbation_type == 'identity':
            self.bottleneck_gru = IdentityPerturbation()
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

        self.decoder = UnetDecoder()
        self.segmentation_head = SegmentationHead()

    def forward(self, x):
        features = self.encoder(x)

        # Insert ConvGRU at the configured layer
        features[self.insertion_point] = self.bottleneck_gru(
            features[self.insertion_point]
        )

        x = self.decoder(features)
        x = self.segmentation_head(x)
        return x

    def forward_without_gru(self, x):
        """Forward pass bypassing the ConvGRU (for validation/comparison)."""
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.segmentation_head(x)
        return x

    def reset_temporal_state(self):
        """Reset GRU hidden state. Call at the start of each sequence."""
        self.bottleneck_gru.reset_hidden_state()

    def detach_temporal_state(self):
        """Detach GRU hidden state from computation graph for truncated BPTT."""
        self.bottleneck_gru.detach_hidden_state()

    def load_from_torchscript(self, torchscript_path):
        """
        Load pretrained weights from the original TorchScript .pt file.

        Maps the TorchScript state_dict keys to our architecture's keys.
        The ConvGRU parameters are left with their random initialization.

        Note: The TorchScript model has nn.Sigmoid() as segmentation_head[2].
        Our model has nn.Identity() there. Since nn.Sigmoid() has no parameters,
        this causes no key mismatch — there's simply no parameter to load for
        that layer.

        Args:
            torchscript_path (str): Path to the TorchScript model file.

        Returns:
            list: Names of parameters that were NOT loaded (i.e., ConvGRU params).
        """
        print(f"Loading pretrained UNet weights from: {torchscript_path}")
        ts_model = torch.jit.load(torchscript_path, map_location='cpu')
        ts_sd = ts_model.state_dict()

        own_sd = self.state_dict()
        loaded = []
        skipped = []

        for key, value in ts_sd.items():
            if key in own_sd:
                if own_sd[key].shape == value.shape:
                    own_sd[key] = value
                    loaded.append(key)
                else:
                    print(f"  SHAPE MISMATCH: {key}: "
                          f"ours={list(own_sd[key].shape)} vs "
                          f"theirs={list(value.shape)}")
                    skipped.append(key)
            else:
                print(f"  NOT FOUND in our model: {key}")
                skipped.append(key)

        # Load the matched parameters
        self.load_state_dict(own_sd, strict=False)

        # Report what was and wasn't loaded
        gru_params = [k for k in own_sd.keys() if 'bottleneck_gru' in k]
        print(f"  Loaded {len(loaded)}/{len(ts_sd)} pretrained parameters")
        print(f"  ConvGRU parameters (randomly initialized): {len(gru_params)}")

        if skipped:
            print(f"  Skipped: {skipped}")

        return gru_params

    def freeze_unet(self):
        """
        Freeze all UNet parameters (encoder + decoder + segmentation head).
        Only the ConvGRU remains trainable.
        """
        for name, param in self.named_parameters():
            if 'bottleneck_gru' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  Frozen: {frozen:,} params (encoder + decoder + head)")
        print(f"  Trainable: {trainable:,} params (ConvGRU)")

    def unfreeze_unet(self, lr_scale=0.01):
        """
        Unfreeze all parameters for end-to-end fine-tuning.
        Returns param groups with different learning rates.
        """
        for param in self.parameters():
            param.requires_grad = True

        gru_params = []
        unet_params = []
        for name, param in self.named_parameters():
            if 'bottleneck_gru' in name:
                gru_params.append(param)
            else:
                unet_params.append(param)

        print(f"  Unfrozen all parameters for end-to-end training")
        print(f"  GRU params: {sum(p.numel() for p in gru_params):,} (full LR)")
        print(f"  UNet params: {sum(p.numel() for p in unet_params):,} ({lr_scale}× LR)")

        return [
            {'params': gru_params},
            {'params': unet_params, 'lr_scale': lr_scale},
        ]
