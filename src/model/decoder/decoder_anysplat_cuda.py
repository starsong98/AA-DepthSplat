from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torchvision

from ..types import Gaussians
# from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
from math import sqrt 
from gsplat import rasterization

#from ...misc.utils import vis_depth_map

# more from depthsplat
from ...dataset import DatasetCfg

# maybe this type will work?
#from ..encoder.common.gaussian_adapter_lpf import Gaussians

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

@dataclass
class DecoderSplattingCUDAAnySplatCfg:
    #name: Literal["splatting_cuda"]
    #background_color: list[float]
    #make_scale_invariant: bool
    name: Literal["splatting_cuda_anysplat"]
    rasterize_mode: Literal["classic", "antialiased"]
    eps2d: float


class DecoderSplattingCUDAAnySplat(Decoder[DecoderSplattingCUDAAnySplatCfg]):
    background_color: Float[Tensor, "3"]
    
    def __init__(
        self,
        cfg: DecoderSplattingCUDAAnySplatCfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        # what they originally did in AnySplat
        #super().__init__(cfg)
        #self.make_scale_invariant = cfg.make_scale_invariant
        #self.register_buffer(
        #    "background_color",
        #    torch.tensor(cfg.background_color, dtype=torch.float32),
        #    persistent=False,
        #)
        # what I am going to do here - match depthsplat
        super().__init__(cfg, dataset_cfg)
        self.make_scale_invariant = True
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        B, V, _, _  = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas = [], [], []
        #xyzs, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        xyzs, opacitys, features = gaussians.means, gaussians.opacities, gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        _, N_gs, _ = xyzs.shape
        scales = getattr(gaussians, "scales", torch.ones([B, N_gs, 3], device=xyzs.device))    # these 2 are gonna be ignored anyway... I hope.
        rotations = getattr(gaussians, "rotations", torch.ones([B, N_gs, 4], device=xyzs.device))  # these 2 are gonna be ignored anyway... I hope.
        covariances = gaussians.covariances
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            covar_i = covariances[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].squeeze().float()
            test_w2c_i = extrinsics[i].float().inverse() # (V, 4, 4)
            test_intr_i_normalized = intrinsics[i].float()
            # Denormalize the intrinsics into standred format
            test_intr_i = test_intr_i_normalized.clone()
            test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = (int(sqrt(feature_i.shape[-2])) - 1)

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            for j in range(V):
                rendering, alpha, _ = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                                test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, sh_degree=sh_degree, 
                                                near_plane=near[i].mean(), far_plane=far[i].mean(),
                                                render_mode="RGB+D", packed=False,
                                                #near_plane=1e-10,
                                                backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                                                radius_clip=0.1,
                                                covars=covar_i,
                                                #rasterize_mode='classic') # (V, H, W, 3) 
                                                #rasterize_mode='classic',
                                                #rasterize_mode='antialiased',
                                                #eps2d=0.1) # (V, H, W, 3)
                                                #eps2d=0.0) # (V, H, W, 3)
                                                rasterize_mode=self.cfg.rasterize_mode,
                                                eps2d=self.cfg.eps2d,
                                            ) # (V, H, W, 3)
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
                rendering_img = rendering_img.clamp(0.0, 1.0)
                rendering_list.append(rendering_img.permute(0, 3, 1, 2))
                rendering_depth_list.append(rendering_depth)
                rendering_alpha_list.append(alpha)
            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze())
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze())
        #return DecoderOutput(torch.stack(rendered_imgs), torch.stack(rendered_depths), torch.stack(rendered_alphas), lod_rendering=None)
        return DecoderOutput(torch.stack(rendered_imgs), torch.stack(rendered_depths))

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        
        return self.rendering_fn(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta, cam_trans_delta)

