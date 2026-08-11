"""
ConvGRU (Convolutional Gated Recurrent Unit) module for adding temporal
memory to the ROI Estimation branch.

The ConvGRU takes the per-frame UNet output and refines it by maintaining a
learned hidden state that carries information across consecutive frames. This
addresses the key weakness of the original single-frame ROI estimator, which
frequently produces flickering, inconsistent masks for tiny objects.

Reference:
    Ballas et al., "Delving Deeper into Convolutional Networks for Learning
    Video Representations", ICLR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    A single Convolutional GRU cell that operates on 2D feature maps.

    Unlike a standard GRU that uses fully-connected layers, this cell uses
    convolutions to preserve spatial structure — critical for segmentation
    masks where the spatial layout of ROIs matters.

    Args:
        input_channels (int): Number of channels in the input tensor.
        hidden_channels (int): Number of channels in the hidden state.
        kernel_size (int): Size of the convolving kernel. Default: 3.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Reset gate: decides how much of the previous hidden state to forget
        self.conv_zr = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,  # z (update) and r (reset) gates combined
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

        # Candidate hidden state
        self.conv_h = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, h_prev):
        """
        Forward pass for a single time step.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            h_prev (torch.Tensor): Previous hidden state of shape (B, C_hid, H, W).

        Returns:
            torch.Tensor: Updated hidden state of shape (B, C_hid, H, W).
        """
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_hid, H, W)

        zr = torch.sigmoid(self.conv_zr(combined))
        z, r = zr.chunk(2, dim=1)  # update gate, reset gate

        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_h(combined_r))

        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new


class TemporalROIRefiner(nn.Module):
    """
    Temporal refinement module that wraps around the frozen UNet ROI estimator.

    Architecture:
        UNet output (1ch) → input projection (1→hidden) �� ConvGRU cell
        → output projection (hidden→1ch) → residual blend → refined mask

    The hidden state persists across frames within a sequence and is reset
    at the start of each new video sequence. This gives the estimator
    learned temporal memory without modifying the frozen UNet weights.

    The module also includes a learnable residual gate (alpha) that blends
    the ConvGRU output with the original UNet output, allowing the model
    to gracefully fall back to single-frame behavior when temporal context
    is not helpful (e.g., first frame of a sequence).

    Args:
        hidden_channels (int): Number of channels in the GRU hidden state.
            Higher = more capacity but slower. Default: 16.
        kernel_size (int): Convolution kernel size for the GRU. Default: 3.
    """

    def __init__(self, hidden_channels=16, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Project 1-channel UNet output to hidden_channels
        self.input_proj = nn.Conv2d(1, hidden_channels, kernel_size=1, bias=True)

        # The core temporal module
        self.gru_cell = ConvGRUCell(hidden_channels, hidden_channels, kernel_size)

        # Project back to 1-channel output
        self.output_proj = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

        # Learnable residual blending gate
        # Initialize to -3.0 so sigmoid(-3)
        self.alpha = nn.Parameter(torch.tensor(-3.0))

        # Hidden state (not a parameter — managed manually)
        self._hidden_state = None

    def reset_hidden_state(self):
        """Reset the hidden state at the start of each new video sequence."""
        self._hidden_state = None

    def detach_hidden_state(self):
        """Detach hidden state from computation graph for truncated BPTT."""
        if self._hidden_state is not None:
            self._hidden_state = self._hidden_state.detach()

    def forward(self, unet_output):
        """
        Refine a single-frame UNet output using temporal context.

        Args:
            unet_output (torch.Tensor): UNet output (post-sigmoid or pre-sigmoid),
                shape (1, 1, H, W).

        Returns:
            torch.Tensor: Temporally refined output, shape (1, 1, H, W).
        """
        B, C, H, W = unet_output.shape

        # Initialize hidden state on first frame or if spatial size changed
        if (self._hidden_state is None
                or self._hidden_state.shape[-2:] != (H, W)):
            self._hidden_state = torch.zeros(
                B, self.hidden_channels, H, W,
                device=unet_output.device, dtype=unet_output.dtype,
            )

        # Project input
        x = self.input_proj(unet_output)  # (B, hidden, H, W)

        # GRU step
        self._hidden_state = self.gru_cell(x, self._hidden_state)

        # Project back to 1 channel
        gru_output = self.output_proj(self._hidden_state)  # (B, 1, H, W)

        # Residual blend: alpha * gru_output + (1-alpha) * unet_output
        alpha = torch.sigmoid(self.alpha)  # constrain to [0, 1]
        refined = alpha * gru_output + (1 - alpha) * unet_output

        return refined
