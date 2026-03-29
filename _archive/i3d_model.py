"""
Inflated 3D ConvNet (I3D) — Inception-v1 RGB variant.
Architecture from "Quo Vadis, Action Recognition?" (Carreira & Zisserman, 2017).
Matches the pytorch_i3d.py used in WLASL (Li et al., 2020) so that pretrained
WLASL checkpoints load without key remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    """MaxPool3d that replicates TensorFlow SAME padding."""

    def _pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        _, _, t, h, w = x.size()
        pt = self._pad(0, t); ph = self._pad(1, h); pw = self._pad(2, w)
        x = F.pad(x, (pw // 2, pw - pw // 2,
                       ph // 2, ph - ph // 2,
                       pt // 2, pt - pt // 2))
        return super().forward(x)


class Unit3D(nn.Module):
    """Conv3d + optional BN + optional activation, with SAME-style padding."""

    def __init__(self, in_channels, out_channels,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 activation_fn=F.relu, use_batch_norm=True, use_bias=False):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size] * 3
        self.stride      = stride      if isinstance(stride,      (list, tuple)) else [stride]      * 3
        self.activation_fn   = activation_fn
        self.use_batch_norm  = use_batch_norm

        self.conv3d = nn.Conv3d(in_channels, out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=0, bias=use_bias)
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01)

    def _pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        _, _, t, h, w = x.size()
        pt = self._pad(0, t); ph = self._pad(1, h); pw = self._pad(2, w)
        x = F.pad(x, (pw // 2, pw - pw // 2,
                       ph // 2, ph - ph // 2,
                       pt // 2, pt - pt // 2))
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """One Inception block with four branches."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # branch 0: 1×1×1
        self.b0 = Unit3D(in_channels, out_channels[0], kernel_size=(1, 1, 1))
        # branch 1: 1×1×1 → 3×3×3
        self.b1a = Unit3D(in_channels,       out_channels[1], kernel_size=(1, 1, 1))
        self.b1b = Unit3D(out_channels[1],   out_channels[2], kernel_size=(3, 3, 3))
        # branch 2: 1×1×1 → 3×3×3
        self.b2a = Unit3D(in_channels,       out_channels[3], kernel_size=(1, 1, 1))
        self.b2b = Unit3D(out_channels[3],   out_channels[4], kernel_size=(3, 3, 3))
        # branch 3: MaxPool → 1×1×1
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_size=(1, 1, 1))

    def forward(self, x):
        return torch.cat([self.b0(x),
                          self.b1b(self.b1a(x)),
                          self.b2b(self.b2a(x)),
                          self.b3b(self.b3a(x))], dim=1)


class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D (RGB stream).

    Input:  (B, 3, T, 224, 224)  — T ≥ 16 recommended
    Output: (B, num_classes)      — averaged over temporal dimension
    """

    def __init__(self, num_classes=400, dropout_keep_prob=0.5, in_channels=3):
        super().__init__()
        self.num_classes = num_classes

        # ---- Stem ----
        self.Conv3d_1a_7x7 = Unit3D(in_channels, 64,  kernel_size=(7, 7, 7), stride=(2, 2, 2))
        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.Conv3d_2b_1x1   = Unit3D(64,  64,  kernel_size=(1, 1, 1))
        self.Conv3d_2c_3x3   = Unit3D(64,  192, kernel_size=(3, 3, 3))
        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)

        # ---- Inception stack ----
        # in=192 → out=256
        self.Mixed_3b = InceptionModule(192,  [64,  96,  128, 16,  32,  32])
        # in=256 → out=480
        self.Mixed_3c = InceptionModule(256,  [128, 128, 192, 32,  96,  64])
        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)

        # in=480 → out=512
        self.Mixed_4b = InceptionModule(480,  [192, 96,  208, 16,  48,  64])
        # in=512 → out=512
        self.Mixed_4c = InceptionModule(512,  [160, 112, 224, 24,  64,  64])
        # in=512 → out=512
        self.Mixed_4d = InceptionModule(512,  [128, 128, 256, 24,  64,  64])
        # in=512 → out=528
        self.Mixed_4e = InceptionModule(512,  [112, 144, 288, 32,  64,  64])
        # in=528 → out=832
        self.Mixed_4f = InceptionModule(528,  [256, 160, 320, 32,  128, 128])
        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)

        # in=832 → out=832
        self.Mixed_5b = InceptionModule(832,  [256, 160, 320, 32,  128, 128])
        # in=832 → out=1024
        self.Mixed_5c = InceptionModule(832,  [384, 192, 384, 48,  128, 128])

        # ---- Classifier head ----
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout  = nn.Dropout(p=1.0 - dropout_keep_prob)
        # 1×1×1 conv acts as a linear layer over 1024 channels
        self.logits   = Unit3D(1024, num_classes, kernel_size=(1, 1, 1),
                               activation_fn=None, use_batch_norm=False, use_bias=True)

    def replace_logits(self, num_classes):
        """Swap the classification head (e.g., for fine-tuning)."""
        self.num_classes = num_classes
        self.logits = Unit3D(1024, num_classes, kernel_size=(1, 1, 1),
                             activation_fn=None, use_batch_norm=False, use_bias=True)

    def forward(self, x):
        # Stem
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)

        # Inception stack
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        # Head: spatial squeeze then temporal average
        x = self.avg_pool(x)           # (B, 1024, T', 1, 1)
        x = self.dropout(x)
        x = self.logits(x)             # (B, num_classes, T', 1, 1)
        x = x.squeeze(4).squeeze(3)    # (B, num_classes, T')
        return x.mean(dim=2)           # (B, num_classes)


if __name__ == "__main__":
    import time

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing InceptionI3d on {device}")

    model = InceptionI3d(num_classes=100).to(device).eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    # Forward pass with a 16-frame clip
    dummy = torch.randn(1, 3, 16, 224, 224, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(dummy)
    elapsed = time.perf_counter() - t0

    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(logits.shape)}")   # (1, 100)
    print(f"Inference time: {elapsed * 1000:.0f} ms")
    print("OK")
