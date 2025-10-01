from .model_wrapper import *

from ..dataset.shims.crop_shim import rescale
from ..dataset.multiscale_data_module import get_data_shim_scaled

import pandas as pd

class ModelWrapper4(ModelWrapper):
    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__(
            optimizer_cfg,
            test_cfg,
            train_cfg,
            encoder,
            encoder_visualizer,
            decoder,
            losses,
            step_tracker,
            eval_data_cfg
        )
        self.data_shim_2 = get_data_shim_scaled(self.encoder, scaling_factor=2)  # for handling the 960p split
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch, batch_960p = batch["480p"], batch["960p"]
        assert batch["scene"] == batch_960p["scene"]
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        #batch_960p: BatchedExample = self.data_shim(batch_960p)
        batch_960p: BatchedExample = self.data_shim_2(batch_960p)

        pred_depths = None

        # debugging camera pose misalignment between resolution splits
        # TODO I should totally make this a separate setting someday
        #camera_poses = batch["target"]["extrinsics"]
        #camera_poses_960p = batch_960p["target"]["extrinsics"]
        #intrinsics_debug= batch["target"]["intrinsics"]
        #intrinsics_debug_960p = batch_960p["target"]["intrinsics"]
        #
        #scene = batch["scene"][0]
        #if not torch.equal(camera_poses, camera_poses_960p):
        #    print(f"Scene {scene}: Target view extrinsics misaligned")
        #if not torch.equal(intrinsics_debug, intrinsics_debug_960p):
        #    print(f"Scene {scene}: Target view intrinsics misaligned")
        #return


        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )

            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]
                gaussians = gaussians["gaussians"]

        # save gaussians
        if self.test_cfg.save_gaussian:
            scene = batch["scene"][0]
            save_path = Path(get_cfg()['output_dir']) / 'gaussians' / (scene + '.ply')
            save_gaussian_ply(gaussians, visualization_dump, batch, save_path)

        if not self.train_cfg.forward_depth_only:
            with self.benchmarker.time("decoder", num_calls=v):

                camera_poses = batch["target"]["extrinsics"]
                #camera_poses_960p = batch_960p["target"]["extrinsics"]
                #intrinsics_debug= batch["target"]["intrinsics"]
                #intrinsics_debug_960p = batch_960p["target"]["intrinsics"]
                #
                ## debugging camera pose misalignment between resolution splits
                #if not torch.equal(camera_poses, camera_poses_960p):
                #    print(f"Scene {scene}: Target view extrinsics misaligned")
                #if not torch.equal(intrinsics_debug, intrinsics_debug_960p):
                #    print(f"Scene {scene}: Target view intrinsics misaligned")
                #return

                if self.test_cfg.stablize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)

                    # same for 960p cameras
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)

                # chunked renders
                if self.test_cfg.render_chunk_size is not None:
                    chunk_size = self.test_cfg.render_chunk_size
                    num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                    output = None
                    for i in range(num_chunks):
                        start = chunk_size * i
                        end = chunk_size * (i + 1)

                        render_intrinsics = batch["target"]["intrinsics"]
                        render_near = batch["target"]["near"]
                        render_far = batch["target"]["far"]

                        curr_output = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h, w),
                            #depth_mode=None,
                            depth_mode=self.test_cfg.depth_mode,
                        )

                        # 1/2 scale outputs
                        curr_output_1_2 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h//2, w//2),
                            #depth_mode=None,
                            depth_mode=self.test_cfg.depth_mode,
                        )

                        # 1/4 scale outputs
                        curr_output_1_4 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h//4, w//4),
                            #depth_mode=None,
                            depth_mode=self.test_cfg.depth_mode,
                        )

                        # 2x scale outputs
                        curr_output_2 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h*2, w*2),
                            #depth_mode=None,
                            depth_mode=self.test_cfg.depth_mode,
                        )

                        # 4x scale outputs
                        curr_output_4 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h*4, w*4),
                            #depth_mode=None,
                            depth_mode=self.test_cfg.depth_mode,
                        )

                        if i == 0:
                            output = curr_output
                            output_2 = curr_output_2
                            output_4 = curr_output_4
                            output_1_2 = curr_output_1_2
                            output_1_4 = curr_output_1_4

                        else:
                            # ignore depth
                            output.color = torch.cat(
                                (output.color, curr_output.color), dim=1
                            )
                            output_2.color = torch.cat(
                                (output_2.color, curr_output_2.color), dim=1
                            )
                            output_4.color = torch.cat(
                                (output_4.color, curr_output_4.color), dim=1
                            )
                            output_1_2.color = torch.cat(
                                (output_1_2.color, curr_output_1_2.color), dim=1
                            )
                            output_1_4.color = torch.cat(
                                (output_1_4.color, curr_output_1_4.color), dim=1
                            )

                # non-chunked renders
                else:
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        #depth_mode=None,
                        depth_mode=self.test_cfg.depth_mode,
                    )
                    # 1/2 scale outputs
                    output_1_2 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h//2, w//2),
                        #depth_mode=None,
                        depth_mode=self.test_cfg.depth_mode,
                    )
                    # 1/4 scale outputs
                    output_1_4 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h//4, w//4),
                        #depth_mode=None,
                        depth_mode=self.test_cfg.depth_mode,
                    )
                    # 2x scale outputs
                    output_2 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h*2, w*2),
                        #depth_mode=None,
                        depth_mode=self.test_cfg.depth_mode,
                    )
                    # 4x scale outputs
                    output_4 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h*4, w*4),
                        #depth_mode=None,
                        depth_mode=self.test_cfg.depth_mode,
                    )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            if self.train_cfg.forward_depth_only:
                depth = pred_depths[0].cpu().detach()  # [V, H, W]
            else:
                depth = (
                    visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
                )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            for idx, depth_i in zip(index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.detach().cpu().numpy()
                    save_path = path / "images" / scene / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene / "depth" /  f"img_depth_{scene}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

            if self.train_cfg.forward_depth_only:
                return

        # output rendered images
        images_prob = output.color[0]
        #rgb_gt = batch["target"]["image"][0]    # [V, 3, H, W]
        images_1_2_prob = output_1_2.color[0]   # 1/2 scale images
        images_1_4_prob = output_1_4.color[0]   # 1/4 scale images
        images_2_prob = output_2.color[0]   # 2x scale images
        images_4_prob = output_4.color[0]   # 4x scale images

        # multiscale gts/pseudo-GTs
        # referring to `src/dataset/shims/crop_shim.py`>L93 to prepare multi-scale 'GT's.
        # TODO might have to doubt this too later
        rgb_gt = batch["target"]["image"][0]    # [V, 3, H, W]
        rgb_gt_1_2 = torch.stack([rescale(rgb_gt_one, (h//2, w//2)) for rgb_gt_one in rgb_gt])  # 1/2 scale GTs
        rgb_gt_1_4 = torch.stack([rescale(rgb_gt_one, (h//4, w//4)) for rgb_gt_one in rgb_gt])  # 1/4 scale GTs
        rgb_gt_2 = torch.stack([rescale(rgb_gt_one, (h*2, w*2)) for rgb_gt_one in rgb_gt])      # 2x scale GTs
        rgb_gt_2 = batch_960p["target"]["image"][0] # 2x scale GTs; [V, 3, 2H, 2W]
        rgb_gt_4 = torch.stack([rescale(rgb_gt_2_one, (h*4, w*4)) for rgb_gt_2_one in rgb_gt_2])      # 4x scale pseudo GTs
        
        # Color-map the result. Taken from the render_video_generic() function.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO depth rendered images
        if self.test_cfg.depth_mode is not None:
            depths_prob = output.depth[0]   # 1x scale predicted depths
            depths_1_2_prob = output_1_2.depth[0]   # 1/2 scale predicted depths
            depths_1_4_prob = output_1_4.depth[0]   # 1/4 scale predicted depths
            depths_2_prob = output_2.depth[0]   # 2x scale predicted depths
            depths_4_prob = output_4.depth[0]   # 4x scale predicted depths

        # Save images.
        if self.test_cfg.save_image:
            # original settings - single scale saving
            #if self.test_cfg.save_gt_image:
            #    for index, color, gt in zip(
            #        batch["target"]["index"][0], images_prob, rgb_gt
            #    ):
            #        save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
            #        save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
            #else:
            #    for index, color in zip(batch["target"]["index"][0], images_prob):
            #        save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
            # modified settings - multi scale saving
            if self.test_cfg.save_gt_image:
                for index, color, color_2, color_4, color_1_2, color_1_4, gt, gt_2, gt_4, gt_1_2, gt_1_4 in zip(
                    batch["target"]["index"][0], 
                    images_prob, images_2_prob, images_4_prob, images_1_2_prob, images_1_4_prob,
                    rgb_gt, rgb_gt_2, rgb_gt_4, rgb_gt_1_2, rgb_gt_1_4
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
                    if self.test_cfg.save_image_upsampled:
                        save_image(color_2, path / "images" / scene / f"color_2/{index:0>6}.png")
                        save_image(color_4, path / "images" / scene / f"color_4/{index:0>6}.png")
                    save_image(color_1_2, path / "images" / scene / f"color_1_2/{index:0>6}.png")
                    save_image(color_1_4, path / "images" / scene / f"color_1_4/{index:0>6}.png")
                    if self.test_cfg.save_image_upsampled:
                        save_image(gt_2, path / "images" / scene / f"color_2/{index:0>6}_gt.png")
                        save_image(gt_4, path / "images" / scene / f"color_4/{index:0>6}_gt.png")
                    save_image(gt_1_2, path / "images" / scene / f"color_1_2/{index:0>6}_gt.png")
                    save_image(gt_1_4, path / "images" / scene / f"color_1_4/{index:0>6}_gt.png")
            else:
                for index, color, color_2, color_4, color_1_2, color_1_4 in zip(
                    batch["target"]["index"][0],
                    images_prob, images_2_prob, images_4_prob, images_1_2_prob, images_1_4_prob
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    if self.test_cfg.save_image_upsampled:
                        save_image(color_2, path / "images" / scene / f"color_2/{index:0>6}.png")
                        save_image(color_4, path / "images" / scene / f"color_4/{index:0>6}.png")
                    save_image(color_1_2, path / "images" / scene / f"color_1_2/{index:0>6}.png")
                    save_image(color_1_4, path / "images" / scene / f"color_1_4/{index:0>6}.png")
            
            if self.test_cfg.save_grid_comparisons:
                # reference: take a page out of validation loop
                comparison = hcat(
                    add_label(vcat(*batch["context"]["image"][0]), "Context"),
                    add_label(vcat(*rgb_gt), "1x Target (Ground Truth)"),
                    add_label(vcat(*images_prob), "1x Target (Prediction)"),
                )
                save_image(comparison, path / "images" / scene / f"grid/grid_vis.png")

                if self.test_cfg.save_grid_comparisons_downsampled:
                    comparison_1_2x = hcat(
                        add_label(vcat(*rgb_gt_1_2), "1/2x Target (Pseudo-GT)"),
                        add_label(vcat(*images_1_2_prob), "1/2x Target (Rendered)"),
                    )
                    save_image(comparison_1_2x, path / "images" / scene / f"grid/grid_vis_1_2x.png")

                    comparison_1_4x = hcat(
                        add_label(vcat(*rgb_gt_1_4), "1/4x PseudoGT", font_size=12),
                        add_label(vcat(*images_1_4_prob), "1/4x Render", font_size=12),
                    )
                    save_image(comparison_1_4x, path / "images" / scene / f"grid/grid_vis_1_4x.png")
                
                if self.test_cfg.save_grid_comparisons_upsampled:
                    comparison_2x = hcat(
                        add_label(vcat(*rgb_gt_2), "2x Target (Ground Truth)"),
                        add_label(vcat(*images_2_prob), "2x Target (Rendered)"),
                    )
                    save_image(comparison_2x, path / "images" / scene / f"grid/grid_vis_2x.png")

                    comparison_4x = hcat(
                        add_label(vcat(*rgb_gt_4), "4x Target (Pseudo-Ground Truth; 2x LANCZOS Upsampled from 2x GT)"),
                        add_label(vcat(*images_4_prob), "4x Target (Rendered)"),
                    )
                    save_image(comparison_4x, path / "images" / scene / f"grid/grid_vis_4x.png")


                if self.test_cfg.depth_mode is not None:
                    comparison_depth = hcat(
                        add_label(vcat(*batch["context"]["image"][0]), "Context"),
                        add_label(vcat(*depth_map(depth)), "Context GS Mean Depth"),
                        add_label(vcat(*rgb_gt), "1x Target (Ground Truth)"),
                        add_label(vcat(*images_prob), "1x Target (Prediction)"),
                        add_label(vcat(*depth_map(depths_prob)), "1x Target GS Rendered Depth"),
                    )
                    save_image(comparison_depth, path / "images" / scene / f"grid/grid_vis_depth.png")

                    if self.test_cfg.save_grid_comparisons_downsampled:
                        comparison_depth_1_2x = hcat(
                            add_label(vcat(*rgb_gt_1_2), "1/2x Target (Pseudo-GT)"),
                            add_label(vcat(*images_1_2_prob), "1/2x Target (Rendered)"),
                            add_label(vcat(*depth_map(depths_1_2_prob)), "1/2x Rendered Depth"),
                        )
                        save_image(comparison_depth_1_2x, path / "images" / scene / f"grid/grid_vis_depth_1_2x.png")

                        comparison_depth_1_4x = hcat(
                            add_label(vcat(*rgb_gt_1_4), "1/4x PseudoGT", font_size=12),
                            add_label(vcat(*images_1_4_prob), "1/4x Render", font_size=12),
                            add_label(vcat(*depth_map(depths_1_4_prob)), "1/4x Depth", font_size=12),
                        )
                        save_image(comparison_depth_1_4x, path / "images" / scene / f"grid/grid_vis_depth_1_4x.png")
                    
                    if self.test_cfg.save_grid_comparisons_upsampled:
                        comparison_depth_2x = hcat(
                            add_label(vcat(*rgb_gt_2), "2x Target (Ground Truth)"),
                            add_label(vcat(*images_2_prob), "2x Target (Rendered)"),
                            add_label(vcat(*depth_map(depths_2_prob)), "2x Rendered Depth"),
                        )
                        save_image(comparison_depth_2x, path / "images" / scene / f"grid/grid_vis_depth_2x.png")

                        comparison_depth_4x = hcat(
                            add_label(vcat(*rgb_gt_4), "4x Target (Pseudo-Ground Truth; 2x LANCZOS Upsampled from 2x GT)"),
                            add_label(vcat(*images_4_prob), "4x Target (Rendered)"),
                            add_label(vcat(*depth_map(depths_4_prob)), "4x Rendered Depth"),
                        )
                        save_image(comparison_depth_4x, path / "images" / scene / f"grid/grid_vis_depth_4x.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            # original scale
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene}_frame_{frame_str}.mp4",
            )
            # 2x scale
            save_video(
                [a for a in images_2_prob],
                path / "videos_2" / f"{scene}_frame_{frame_str}.mp4",
            )
            # 4x scale
            save_video(
                [a for a in images_4_prob],
                path / "videos_4" / f"{scene}_frame_{frame_str}.mp4",
            )
            # 1/2 scale
            save_video(
                [a for a in images_1_2_prob],
                path / "videos_1_2" / f"{scene}_frame_{frame_str}.mp4",
            )
            # 1/4 scale
            save_video(
                [a for a in images_1_4_prob],
                path / "videos_1_4" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            # scene name - will be used for full test set csv later
            # so this one causes problems down the road due to averaging operations. should make it skip that.
            if f"scene" not in self.test_step_outputs:
                self.test_step_outputs[f"scene"] = []
            self.test_step_outputs[f"scene"].append(scene)

            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if not self.train_cfg.forward_depth_only:
                rgb = images_prob

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                    self.test_step_outputs[f"psnr_1_2"] = []
                    self.test_step_outputs[f"psnr_1_4"] = []
                    if not self.test_cfg.skip_upsampled_scores:
                        self.test_step_outputs[f"psnr_2"] = []
                        #self.test_step_outputs[f"psnr_4"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                    self.test_step_outputs[f"ssim_1_2"] = []
                    self.test_step_outputs[f"ssim_1_4"] = []
                    if not self.test_cfg.skip_upsampled_scores:
                        self.test_step_outputs[f"ssim_2"] = []
                        #self.test_step_outputs[f"ssim_4"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []
                    self.test_step_outputs[f"lpips_1_2"] = []
                    self.test_step_outputs[f"lpips_1_4"] = []
                    if not self.test_cfg.skip_upsampled_scores:
                        self.test_step_outputs[f"lpips_2"] = []
                        #self.test_step_outputs[f"lpips_4"] = []

                # native scale metrics - original
                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )
                # 1/2 scale metrics
                self.test_step_outputs[f"psnr_1_2"].append(
                    compute_psnr(rgb_gt_1_2, images_1_2_prob).mean().item()
                )
                self.test_step_outputs[f"ssim_1_2"].append(
                    compute_ssim(rgb_gt_1_2, images_1_2_prob).mean().item()
                )
                self.test_step_outputs[f"lpips_1_2"].append(
                    compute_lpips(rgb_gt_1_2, images_1_2_prob).mean().item()
                )
                # 1/4 scale metrics
                self.test_step_outputs[f"psnr_1_4"].append(
                    compute_psnr(rgb_gt_1_4, images_1_4_prob).mean().item()
                )
                self.test_step_outputs[f"ssim_1_4"].append(
                    compute_ssim(rgb_gt_1_4, images_1_4_prob).mean().item()
                )
                self.test_step_outputs[f"lpips_1_4"].append(
                    compute_lpips(rgb_gt_1_4, images_1_4_prob).mean().item()
                )
                if not self.test_cfg.skip_upsampled_scores:
                    # 2x scale metrics
                    self.test_step_outputs[f"psnr_2"].append(
                        compute_psnr(rgb_gt_2, images_2_prob).mean().item()
                    )
                    self.test_step_outputs[f"ssim_2"].append(
                        compute_ssim(rgb_gt_2, images_2_prob).mean().item()
                    )
                    self.test_step_outputs[f"lpips_2"].append(
                        compute_lpips(rgb_gt_2, images_2_prob).mean().item()
                    )
                    # 4x scale metrics
                    #self.test_step_outputs[f"psnr_4"].append(
                    #    compute_psnr(rgb_gt_4, images_4_prob).mean().item()
                    #)
                    #self.test_step_outputs[f"ssim_4"].append(
                    #    compute_ssim(rgb_gt_4, images_4_prob).mean().item()
                    #)
                    #self.test_step_outputs[f"lpips_4"].append(
                    #    compute_lpips(rgb_gt_4, images_4_prob).mean().item()
                    #)
            
            # per-scene metrics
            current_scores = {}
            for metric_name, metric_scores in self.test_step_outputs.items():
                current_scores[metric_name] = metric_scores[-1]
            with (path / "images" / scene / "metrics.json").open("w") as f:
                json.dump(current_scores, f)

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            # Saving table for the entire dataset all at once
            # Option 1: Using pandas (handles mixed data types automatically)
            df = pd.DataFrame(self.test_step_outputs)  # Single row, each key becomes a column
            df.to_csv(out_dir / "scores_all.csv", index=True)
            # for average metric computation later
            self.test_step_outputs.pop("scene", None)

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        # native resolution video (same as original)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        # 1/4 res.
        output_prob_1_4 = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h//4, w//4), "depth"
        )
        # 2x res.
        output_prob_2 = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h*2, w*2), "depth"
        )
        # 4x res.
        #output_prob_4 = self.decoder.forward(
        #    gaussians_prob, extrinsics, intrinsics, near, far, (h*4, w*4), "depth"
        #)

        # native res.
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        images_prob_1_4 = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob_1_4.color[0], depth_map(output_prob_1_4.depth[0]))
        ]
        images_prob_2 = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob_2.color[0], depth_map(output_prob_2.depth[0]))
        ]
        #images_prob_4 = [
        #    vcat(rgb, depth)
        #    for rgb, depth in zip(output_prob_4.color[0], depth_map(output_prob_4.depth[0]))
        #]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]
        images_1_4 = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction (1/4x)", font_size=12),
                )
            )
            for image_prob, _ in zip(images_prob_1_4, images_prob_1_4)
        ]
        images_2 = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction (2x)", font_size=36),
                )
            )
            for image_prob, _ in zip(images_prob_2, images_prob_2)
        ]
        #images_4 = [
        #    add_border(
        #        hcat(
        #            add_label(image_prob, "Prediction (4x)", font_size=72),
        #        )
        #    )
        #    for image_prob, _ in zip(images_prob_4, images_prob_4)
        #]

        video = torch.stack(images)
        video_1_4 = torch.stack(images_1_4)
        video_2 = torch.stack(images_2)
        #video_4 = torch.stack(images_4)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        video_1_4 = (video_1_4.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        video_2 = (video_2.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        #video_4 = (video_4.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
            video_1_4 = pack([video_1_4, video_1_4[::-1][1:-1]], "* c h w")[0]
            video_2 = pack([video_2, video_2[::-1][1:-1]], "* c h w")[0]
            #video_4 = pack([video_4, video_4[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4"), # native res. - original
            f"video_1_4/{name}": wandb.Video(video_1_4[None], fps=30, format="mp4"),
            f"video_2/{name}": wandb.Video(video_2[None], fps=30, format="mp4")
            #f"video/{name}_4": wandb.Video(video_4[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )
    
    def configure_optimizers(self):
        if getattr(self.optimizer_cfg, "train_gs_head_only", False):
            print("Finetuning only the GS head params")
            gs_head_params = []
            updated_names = []
            frozen_names = []

            for name, param in self.named_parameters():
                if "feature_upsampler" in name:
                    gs_head_params.append(param)
                    updated_names.append(name)
                    #print(f"Will be added to optimizer - {name}")
                elif "gaussian" in name:
                    gs_head_params.append(param)
                    updated_names.append(name)
                    #print(f"Will be added to optimizer - {name}")
                else:
                    param.requires_grad = False
                    frozen_names.append(name)
            
            print("The following params will be updated during training:")
            for name in updated_names:
                print(f"\t{name}")
            print("")
            print("The following params will be frozen during training:")
            for name in frozen_names:
                print(f"\t{name}")
            
            optimizer = torch.optim.AdamW(
                gs_head_params,
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )
            
        else:
            # full model is to be trained/finetuned
            if self.optimizer_cfg.lr_gshead > 0:
                # GS heads get their own learning rates
                if self.optimizer_cfg.lr_monodepth > 0:
                    # monocular branch gets its own learning rate
                    pretrained_params = []
                    new_params = []
                    gs_head_params = []

                    for name, param in self.named_parameters():
                        if "pretrained" in name:
                            pretrained_params.append(param)
                        elif "gaussian" in name:
                            gs_head_params.append(param)
                        else:
                            new_params.append(param)

                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": pretrained_params,
                                "lr": self.optimizer_cfg.lr_monodepth,
                            },
                            {
                                "params": gs_head_params,
                                "lr": self.optimizer_cfg.lr_gshead,
                            },
                            {"params": new_params, "lr": self.optimizer_cfg.lr},
                        ],
                        weight_decay=self.optimizer_cfg.weight_decay,
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        [
                            self.optimizer_cfg.lr_monodepth,
                            self.optimizer_cfg.lr_gshead,
                            self.optimizer_cfg.lr
                        ],
                        self.trainer.max_steps + 10,
                        pct_start=0.01,
                        cycle_momentum=False,
                        anneal_strategy="cos",
                    )

                else:
                    # the entire depth backbone (i.e., everything except for the GS head) gets the same lr
                    depth_params = []
                    gs_head_params = []

                    for name, param in self.named_parameters():
                        if "gaussian" in name:
                            gs_head_params.append(param)
                        else:
                            depth_params.append(param)

                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": gs_head_params,
                                "lr": self.optimizer_cfg.lr_gshead,
                            },
                            {"params": depth_params, "lr": self.optimizer_cfg.lr},
                        ],
                        weight_decay=self.optimizer_cfg.weight_decay,
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        [
                            self.optimizer_cfg.lr_gshead,
                            self.optimizer_cfg.lr
                        ],
                        self.trainer.max_steps + 10,
                        pct_start=0.01,
                        cycle_momentum=False,
                        anneal_strategy="cos",
                    )
            else:
                # vanilla cases
                if self.optimizer_cfg.lr_monodepth > 0:
                    pretrained_params = []
                    new_params = []

                    for name, param in self.named_parameters():
                        if "pretrained" in name:
                            pretrained_params.append(param)
                        else:
                            new_params.append(param)

                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": pretrained_params,
                                "lr": self.optimizer_cfg.lr_monodepth,
                            },
                            {"params": new_params, "lr": self.optimizer_cfg.lr},
                        ],
                        weight_decay=self.optimizer_cfg.weight_decay,
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                        self.trainer.max_steps + 10,
                        pct_start=0.01,
                        cycle_momentum=False,
                        anneal_strategy="cos",
                    )

                else:
                    optimizer = optim.AdamW(
                        self.parameters(),
                        lr=self.optimizer_cfg.lr,
                        weight_decay=self.optimizer_cfg.weight_decay,
                    )

                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        self.optimizer_cfg.lr,
                        self.trainer.max_steps + 10,
                        pct_start=0.01,
                        cycle_momentum=False,
                        anneal_strategy="cos",
                    )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }