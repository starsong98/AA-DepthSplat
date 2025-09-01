from .gaussian_adapter import *

from ....geometry.projection import project, transform_world2cam, homogenize_points

class GaussianAdapterLPF(GaussianAdapter):
    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__(cfg)
    
    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        # split the raw gaussians
        # [B, V, H*W, 1,1, 3], [B, V, H*W, 1,1, 4], [B, V, H*W, 1,1, 27]
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        # scales mapping
        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)
        
        # Compute Gaussian means - do this earlier
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]    # 3D world coordinates; [B, V, HW, 1, 1, 3]

        # TODO compute 3D filter sizes for each GS
        filter_3D = self.compute_3D_filters(extrinsics, intrinsics, means, input_images, depths)

        # TODO new scales & opacities
        opacities, scales = self.get_opacity_scaling_with_3D_filter(opacities, scales, filter_3D)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        #origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        #means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )
    
    @torch.no_grad()
    def compute_3D_filters(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],   # [B, V, 1, 1, 1, 4, 4]
        intrinsics: Float[Tensor, "*#batch 3 3"],   # [B, V, 1, 1, 1, 3, 3]
        xyz: Float[Tensor, "*#batch 3"],          # [B, V, H*W, 1, 1, 3]
        input_images: Tensor,   # [B, V, 3, H, W]
        depths: Float[Tensor, "*#batch"],   # [B, V, H*W, 1, 1]
    ) -> Tensor:
        # Based on:
        # src/model/encoder/unimatch/matching.py > warp_with_pose_depth_candidates()
        # https://github1s.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py#L142-L191

        #xyz = means
        #distance = torch.ones((xyz.shape[:-1]), device=xyz.device) * 100000.0
        # initialize with per-view depth instead of some arbitrary value
        distance = depths.clone()

        valid_points = torch.zeros((xyz.shape[:-1]), device=xyz.device, dtype=torch.bool)

        b, v, _, ori_h, ori_w = input_images.shape

        # we should use the focal length of the highest resolution camera
        focal_length = torch.zeros([b, v, 1, 1, 1], device=xyz.device)

        # loop over all views
        for i in range(v):
            # retrieve the i-th context view intrinsics/extrinsics
            # and change their shapes to fit what we want to do with them
            extrinsics_i = extrinsics[:, i].unsqueeze(1).repeat(1, v, 1, 1, 1, 1, 1)
            intrinsics_i = intrinsics[:, i].unsqueeze(1).repeat(1, v, 1, 1, 1, 1, 1)

            # transform to i-th context view camera space
            xyz_homo = homogenize_points(xyz)
            xyz_cam = transform_world2cam(xyz_homo, extrinsics_i)
            x_cam, y_cam, z_cam = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
            #valid_depth = z_cam > 0.2
            valid_depth = z_cam > 0.1
            z_cam = torch.clamp(z_cam, min=0.001)   # hard coded, following mip-splatting
            
            # project to screen space
            # call the project() function
            xy_img, in_front_of_camera = project(xyz, extrinsics_i, intrinsics_i)
            x_img, y_img = xy_img[..., 0], xy_img[..., 1]

            # screen space filtering - following mip-splatting
            in_screen = torch.logical_and(
                torch.logical_and(x_img >= -0.15, x_img <= 1.15),
                torch.logical_and(y_img >= -0.15, y_img <= 1.15),
            )
            in_native_screen = in_screen[:, i]
            in_native_screen_ratio = in_native_screen.float().mean()    # this needs to be 1.0

            # visibility in i-th context view
            valid = torch.logical_and(valid_depth, in_screen)

            distance[valid] = torch.min(distance[valid], z_cam[valid])
            valid_points = torch.logical_or(valid_points, valid)

            # update focal length
            focal_x = intrinsics_i[:, ..., 0, 0] * ori_w    # [B]
            focal_length = torch.max(focal_x, focal_length)
        
        # this should not have to be done, but it would REALLY suck if this somehow caused a crash
        #distance[~valid_points] = distance[valid_points].max()

        # denormalized_focal_length unit: [pixels]
        # depth unit: [world_units] (whatever the world coordinates use)
        # (depth / denormalized_focal_length) unit: [world_units / pixels]
        # (3D_filter_stdev) = (depth / denormalized_focal_length) * (0.2[pixels^2] ** 0.5) unit --> [world_units]
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        
        return filter_3D
    
    def get_opacity_scaling_with_3D_filter(
        self,
        opacity: Float[Tensor, "*#batch"],      # [B, V, H*W, 1, 1]
        scales: Float[Tensor, "*#batch 3"],     # [B, V, H*W, 1, 1, 3]
        filter_3D: Float[Tensor, "*#batch"],    # [B, V, H*W, 1, 1]
    ):
        # Based on:
        # https://github1s.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py#L99-L105
        # https://github1s.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py#L125-L137
        # TODO actual debugging needed
        # opacity activation (sigmoid) was already applied earlier in the model - no need

        # now apply 3d filter
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=-1)

        scales_after_square = scales_square + torch.square(filter_3D)[..., None]
        det2 = scales_after_square.prod(dim=-1)

        # modified opacity
        coef = torch.sqrt(det1 / det2)
        opacity_filtered = opacity * coef   # [B, V, H*W, 1, 1]

        # modified scales
        scales_filtered = torch.sqrt(scales_after_square)   # [B, V, H*W, 1, 1, 3]

        return opacity_filtered, scales_filtered

    #def get_opacity_with_3D_filter(
    #    self,
    #    opacity,
    #    3D_filters
    #) -> Tensor:
    #    # Based on:
    #    # https://github1s.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py#L125-L137
    #    # TODO
    #    return
    #
    #def get_scaling_with_3D_filter(self, scales, 3D_filters):
    #    # Based on:
    #    # https://github1s.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py#L99-L105
    #    # TODO
    #    return