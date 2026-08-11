"""
Perturbation modules — ablation controls for the temporal ConvGRU.

`TemporalUnet` (rois/estimator/unet_resnet18.py) imports these four classes at
construction time, so this file **must exist** even when `--perturbation_type gru`
is used and none of them is instantiated.

Each class is a drop-in replacement for `BottleneckConvGRU` at the same encoder
insertion point and answers the question "is the gain caused by temporal memory,
or merely by perturbing the features at that layer?":

    IdentityPerturbation        pass-through; isolates the effect of the
                                Phase-2 joint fine-tuning alone.
    GaussianPerturbation        additive Gaussian noise instead of memory.
    Dropout2dPerturbation       channel dropout instead of memory.
    FrozenRandomGRUPerturbation a ConvGRU with frozen random weights: keeps the
                                architecture and the recurrence, removes the
                                learned content.

All of them expose the interface `TemporalUnet` relies on — `forward`,
`reset_hidden_state`, `detach_hidden_state` — and carry a learnable `alpha`
scalar so that checkpoints stay structurally comparable across ablations.

NOTE: if you already have your own version of this file on the training
machine, keep it. This one is written to match the call sites in
`unet_resnet18.py` and the checkpoint-sniffing logic in
`bottleneck_temporal_estimator.py`, but your trained ablation checkpoints were
produced by yours.
"""

import torch
import torch.nn as nn


class _PerturbationBase(nn.Module):
    """Common interface expected by TemporalUnet.

    The `alpha` parameter is kept even where it is unused so that every
    ablation checkpoint contains the same `bottleneck_gru.alpha` key. The
    estimator uses the presence/absence of other keys to infer which variant
    produced a checkpoint.
    """

    def __init__(self, alpha_init=-3.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def reset_hidden_state(self):
        """No-op for stateless perturbations; overridden where relevant."""
        return None

    def detach_hidden_state(self):
        """No-op for stateless perturbations; overridden where relevant."""
        return None


class IdentityPerturbation(_PerturbationBase):
    """Returns the features unchanged.

    Used to measure how much of the observed change comes from the Phase-2
    joint fine-tuning of the UNet rather than from the temporal block itself.
    """

    def forward(self, x):
        return x


class GaussianPerturbation(_PerturbationBase):
    """Adds zero-mean Gaussian noise to the features during training.

    At inference the module is an identity mapping, so a checkpoint trained
    with this perturbation evaluates as a plain (fine-tuned) UNet.

    Args:
        std (float): standard deviation of the injected noise.
    """

    def __init__(self, std=0.05, alpha_init=-3.0):
        super().__init__(alpha_init=alpha_init)
        self.std = float(std)

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class Dropout2dPerturbation(_PerturbationBase):
    """Applies channel-wise dropout to the features during training.

    Like GaussianPerturbation, this is an identity mapping at inference.

    Args:
        p (float): channel dropout probability.
    """

    def __init__(self, p=0.05, alpha_init=-3.0):
        super().__init__(alpha_init=alpha_init)
        self.p = float(p)
        self.drop = nn.Dropout2d(p=self.p)

    def forward(self, x):
        if self.training and self.p > 0:
            return self.drop(x)
        return x


class FrozenRandomGRUPerturbation(nn.Module):
    """A ConvGRU with randomly initialised, permanently frozen weights.

    Keeps the recurrent architecture and its temporal state, but never learns.
    Only the blend gate `alpha` is trainable, so any measured effect is due to
    the recurrence itself and not to what it learned.

    The inner block is stored as `self.gru`, which is how
    `BottleneckTemporalEstimator` recognises checkpoints of this variant
    (key prefix ``bottleneck_gru.gru.``).

    Args:
        bottleneck_channels (int): channels at the insertion point.
        hidden_channels (int): ConvGRU hidden channels.
        kernel_size (int): ConvGRU kernel size.
        alpha_init (float): pre-sigmoid initial value of the blend gate.
    """

    def __init__(self, bottleneck_channels=512, hidden_channels=64,
                 kernel_size=3, alpha_init=-5.0):
        super().__init__()
        # Imported here rather than at module level: unet_resnet18 imports this
        # file from inside TemporalUnet.__init__, so a top-level import back
        # into it would be circular.
        from .unet_resnet18 import BottleneckConvGRU

        self.gru = BottleneckConvGRU(
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            alpha_init=alpha_init,
        )
        for p in self.gru.parameters():
            p.requires_grad = False

        # Only the outer gate is trainable.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def reset_hidden_state(self):
        self.gru.reset_hidden_state()

    def detach_hidden_state(self):
        self.gru.detach_hidden_state()

    def forward(self, x):
        with torch.no_grad():
            temporal_out = self.gru(x)
        gate = torch.sigmoid(self.alpha)
        return (1 - gate) * x + gate * temporal_out
