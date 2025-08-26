from .model_wrapper import *

from ..dataset.shims.crop_shim import rescale

class ModelWrapper2(ModelWrapper):
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
    
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None

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
                            depth_mode=None,
                        )

                        # 1/2 scale outputs
                        #curr_output_2 = self.decoder.forward(
                        #    gaussians,
                        #    camera_poses[:, start:end],
                        #    render_intrinsics[:, start:end],
                        #    render_near[:, start:end],
                        #    render_far[:, start:end],
                        #    (h//2, w//2),
                        #    depth_mode=None,
                        #)
                        curr_output_1_2 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h//2, w//2),
                            depth_mode=None,
                        )

                        # 1/4 scale outputs
                        #curr_output_4 = self.decoder.forward(
                        #    gaussians,
                        #    camera_poses[:, start:end],
                        #    render_intrinsics[:, start:end],
                        #    render_near[:, start:end],
                        #    render_far[:, start:end],
                        #    (h//4, w//4),
                        #    depth_mode=None,
                        #)
                        curr_output_1_4 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h//4, w//4),
                            depth_mode=None,
                        )

                        # 2x scale outputs
                        curr_output_2 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h*2, w*2),
                            depth_mode=None,
                        )

                        # 4x scale outputs
                        curr_output_4 = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h*4, w*4),
                            depth_mode=None,
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
                        depth_mode=None,
                    )
                    # 1/2 scale outputs
                    output_1_2 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h//2, w//2),
                        depth_mode=None,
                    )
                    # 1/4 scale outputs
                    output_1_4 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h//4, w//4),
                        depth_mode=None,
                    )
                    # 2x scale outputs
                    output_2 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h*2, w*2),
                        depth_mode=None,
                    )
                    # 4x scale outputs
                    output_4 = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h*4, w*4),
                        depth_mode=None,
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

        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]    # [V, 3, H, W]
        images_1_2_prob = output_1_2.color[0]   # 1/2 scale images
        images_1_4_prob = output_1_4.color[0]   # 1/4 scale images
        images_2_prob = output_2.color[0]   # 2x scale images
        images_4_prob = output_4.color[0]   # 4x scale images
        # referring to `src/dataset/shims/crop_shim.py`>L93 to prepare multi-scale 'GT's.
        rgb_gt_1_2 = torch.stack([rescale(rgb_gt_one, (h//2, w//2)) for rgb_gt_one in rgb_gt])  # 1/2 scale GTs
        rgb_gt_1_4 = torch.stack([rescale(rgb_gt_one, (h//4, w//4)) for rgb_gt_one in rgb_gt])  # 1/4 scale GTs
        rgb_gt_2 = torch.stack([rescale(rgb_gt_one, (h*2, w*2)) for rgb_gt_one in rgb_gt])      # 2x scale GTs
        rgb_gt_4 = torch.stack([rescale(rgb_gt_one, (h*4, w*4)) for rgb_gt_one in rgb_gt])      # 4x scale GTs

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
                    #save_image(color_2, path / "images_2" / scene / f"color/{index:0>6}.png")
                    save_image(color_2, path / "images" / scene / f"color_2/{index:0>6}.png")
                    save_image(color_4, path / "images" / scene / f"color_4/{index:0>6}.png")
                    save_image(color_1_2, path / "images" / scene / f"color_1_2/{index:0>6}.png")
                    save_image(color_1_4, path / "images" / scene / f"color_1_4/{index:0>6}.png")
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
                    #save_image(color_2, path / "images_2" / scene / f"color/{index:0>6}.png")
                    save_image(color_2, path / "images" / scene / f"color_2/{index:0>6}.png")
                    save_image(color_4, path / "images" / scene / f"color_4/{index:0>6}.png")
                    save_image(color_1_2, path / "images" / scene / f"color_1_2/{index:0>6}.png")
                    save_image(color_1_4, path / "images" / scene / f"color_1_4/{index:0>6}.png")

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
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if not self.train_cfg.forward_depth_only:
                rgb = images_prob

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                    self.test_step_outputs[f"psnr_2"] = []
                    self.test_step_outputs[f"psnr_4"] = []
                    self.test_step_outputs[f"psnr_1_2"] = []
                    self.test_step_outputs[f"psnr_1_4"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                    self.test_step_outputs[f"ssim_2"] = []
                    self.test_step_outputs[f"ssim_4"] = []
                    self.test_step_outputs[f"ssim_1_2"] = []
                    self.test_step_outputs[f"ssim_1_4"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []
                    self.test_step_outputs[f"lpips_2"] = []
                    self.test_step_outputs[f"lpips_4"] = []
                    self.test_step_outputs[f"lpips_1_2"] = []
                    self.test_step_outputs[f"lpips_1_4"] = []

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
                self.test_step_outputs[f"psnr_4"].append(
                    compute_psnr(rgb_gt_4, images_4_prob).mean().item()
                )
                self.test_step_outputs[f"ssim_4"].append(
                    compute_ssim(rgb_gt_4, images_4_prob).mean().item()
                )
                self.test_step_outputs[f"lpips_4"].append(
                    compute_lpips(rgb_gt_4, images_4_prob).mean().item()
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

