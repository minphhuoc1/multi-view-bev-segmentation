# spatial_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformerFixed(nn.Module):
    """
    Spatial Transformer with fixed projective (homography) transform.
    - input: (B, C, H, W)
    - H: (3, 3) numpy or torch tensor (homography matrix)
    - output: (B, C, out_h, out_w)
    """
    def __init__(self, out_size=None, homography=None):
        super().__init__()
        # out_size có thể None, sẽ set ở forward nếu chưa biết
        if out_size is not None:
            self.out_h, self.out_w = out_size
        else:
            self.out_h, self.out_w = None, None
        if isinstance(homography, torch.Tensor):
            self.register_buffer('H', homography.float())
        else:
            self.register_buffer('H', torch.from_numpy(homography).float())


    def get_grid(self, device):
        assert self.out_h is not None and self.out_w is not None, "out_h, out_w must be set before calling get_grid"

        # Create normalized grid [-1, 1] (out_h, out_w)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, self.out_h, device=device),
            torch.linspace(-1, 1, self.out_w, device=device),
            indexing='ij'
        )
        ones = torch.ones_like(xs)
        grid = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, N)
        # Apply inverse homography to grid
        H_inv = torch.inverse(self.H)
        warped = H_inv @ grid  # (3, N)
        warped = warped / (warped[2:3, :] + 1e-8)
        x_warp = warped[0, :].reshape(self.out_h, self.out_w)
        y_warp = warped[1, :].reshape(self.out_h, self.out_w)
        # Normalize to [-1, 1] for grid_sample
        grid_sample = torch.stack([x_warp, y_warp], dim=-1)  # (out_h, out_w, 2)
        return grid_sample

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, out_h, out_w)
        """
        B, C, H, W = x.shape
        grid = self.get_grid(x.device)  # (out_h, out_w, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, out_h, out_w, 2)
        out = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out