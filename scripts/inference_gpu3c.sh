# render video on re10k (need to have ffmpeg installed)
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

# 2 input views at 256x256 resolutions:
# render video on re10k (need to have ffmpeg installed)
#python -m src.main +experiment=re10k \
#dataset.test_chunk_interval=100 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
#test.save_video=true \
#test.compute_scores=false \
#output_dir=outputs/depthsplat-re10k 2>&1 | tee 20250821_run001.log

#python -m src.main +experiment=re10k \
#dataset.test_chunk_interval=1000 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
#test.save_video=true \
#test.compute_scores=false \
#output_dir=outputs/depthsplat-re10k_256x256_run002 2>&1 | tee outputs/20250821_run002.log

# full official toy subset
#python -m src.main +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
#test.save_video=true \
#test.compute_scores=false \
#output_dir=outputs/depthsplat-re10k_256x256_run002 2>&1 | tee outputs/20250821_run003_subset.log
# so the test split of this thing has 41 samples.
# why does it fall short?

# full official toy subset, highres trials
# apparently this does not work - subset is 360p.
#python -m src.main +experiment=dl3dv \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#dataset.image_shape=[512,960] \
#dataset.ori_image_shape=[720,1280] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=6 \
#dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
#test.save_video=true \
#test.compute_scores=false \
#test.render_chunk_size=10 \
#output_dir=outputs/depthsplat-re10k-512x960_run004 2>&1 | tee outputs/20250821_run004_720p_subset.log

# full official toy subset, single-scale inference single-scale rendering
#python -m src.main +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
#test.save_video=true \
#test.compute_scores=true \
#output_dir=outputs/depthsplat-re10k_256x256_run010_metrics 2>&1 | tee outputs/20250821_run010_subset.log

# one scale inference one scale rendering

# full official toy subset, highres trials
# trying with new 720p subset
#python -m src.main +experiment=dl3dv \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets_extra/re10k_720p_test_subset] \
#dataset.image_shape=[512,960] \
#dataset.ori_image_shape=[720,1280] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=6 \
#dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
#test.save_video=true \
#test.compute_scores=false \
#test.render_chunk_size=10 \
#output_dir=outputs/depthsplat-re10k-512x960_run011 2>&1 | tee outputs/20250821_run011_720p_subset.log

# now to record metrics
#python -m src.main +experiment=dl3dv \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets_extra/re10k_720p_test_subset] \
#dataset.image_shape=[512,960] \
#dataset.ori_image_shape=[720,1280] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=6 \
#dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
#test.save_video=true \
#test.compute_scores=true \
#test.render_chunk_size=10 \
#output_dir=outputs/depthsplat-re10k-512x960_run012 2>&1 | tee outputs/20250821_run012_720p_subset.log

# same checkpoint, full test set, evaluation setting
# now saving 1/4x ~ 4x scale GTs (actually 2x and 4x are not quite gts)
# and metrics on all GTs
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-re10k_256x256_simt_fullset_run006 \
#2>&1 | tee outputs/20250825_run006_simtrender_fullset.log

# acid subset testing...
#python -m src.main_2 +experiment=re10k \
#mode=test \
#dataset.roots=[datasets/acid_subset] \
#dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=2 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#test.save_video=false \
#output_dir=outputs/depthsplat-acid_256x256_simt_subset_run006 \
#2>&1 | tee outputs/20250826_run001_simtrender_subset.log

# acid full set testing...
#python -m src.main_2 +experiment=re10k \
#mode=test \
#dataset.roots=[datasets_extra/acid] \
#dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=2 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#test.save_video=false \
#output_dir=outputs/depthsplat-acid_256x256_simt_fullset_run002 \
#2>&1 | tee outputs/20250826_run002_simtrender_acidfullset.log

# dl3dv high-res video rendering, subset
#python -m src.main_2 +experiment=dl3dv \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets_extra/dl3dv_960p_test_subset] \
#dataset.image_shape=[512,960] \
#dataset.ori_image_shape=[540,960] \
#model.encoder.upsample_factor=8 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=12 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_100_ctx_12v_video.json \
#test.save_video=true \
#test.stablize_camera=true \
#test.compute_scores=false \
#test.render_chunk_size=10 \
#output_dir=outputs/depthsplat-dl3dv-512x960_subset_run002_simt \
#2>&1 | tee outputs/20250826_run002_simtrender_dl3dv960p_ctx12v_subset.log

# dl3dv high-res video rendering, subset
#python -m src.main_2 +experiment=dl3dv \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets_extra/dl3dv_960p_test_subset] \
#dataset.image_shape=[512,960] \
#dataset.ori_image_shape=[540,960] \
#model.encoder.upsample_factor=8 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=12 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_100_ctx_12v_video.json \
#test.save_video=true \
#test.stablize_camera=false \
#test.compute_scores=false \
#test.render_chunk_size=5 \
#output_dir=outputs/depthsplat-dl3dv-512x960_subset_run003_simt \
#2>&1 | tee outputs/20250826_run003_simtrender_dl3dv960p_ctx12v_subset.log

# small model baseline establishment
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#output_dir=outputs/depthsplat-S-re10k_256x256_fullset_run001_simt \
#2>&1 | tee outputs/20250828_run001_simt_re10k_fullset_smallmodel.log

# base model baseline establishment
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitb \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-B-re10k_256x256_fullset_run001_simt \
#2>&1 | tee outputs/20250829_run001_simt_re10k_fullset_basemodel.log

# small model + 3D LPF w/o ft
# ignore this - did not actually give 3D LPF
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-3DLPF-Small-re10k_256x256_simt_fullset_run001 \
#2>&1 | tee outputs/20250901_run001_simtrender_fullset_3DLPF_small.log

# small model + 3D LPF w/o ft
# ignore this - did not actually give 3D LPF
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/20250902_run004_re10k-256x256-simt-fullset_3DLPF-Small \
#2>&1 | tee outputs/20250902_run004_re10k-256x256-simt-fullset_3DLPF-Small.log

# RE10k 256x256, small checkpoint
# DepthSplat + 3D LPF, same rasterizer
# GS head only finetuned, 100k steps x batch size 8 x 2 GPUs
# toy test set
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_005/checkpoints/epoch_12-step_100000.ckpt \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/2025-09-03_run002_re10k-256x256-simt-subset_3DLPF-GSHeadft-Small \
#2>&1 | tee outputs/2025-09-03_run002_re10k-256x256-simt-subset_3DLPF-GSHeadft-Small.log

# full mipsplat + depthsplat finetuned prototype,
# full test set, evaluation setting
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-08_train001_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullft/checkpoints/epoch_12-step_200000.ckpt \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/2025-09-09_test001_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft \
#2>&1 | tee outputs/2025-09-09_test001_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft.log

# full mipsplat + depthsplat full trained prototype,
# full test set, evaluation setting
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/2025-09-12_test001_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft \
#2>&1 | tee outputs/2025-09-12_test001_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft.log

############################################################
# Debugging - trying out various test settings
############################################################

# DL3DV 480p sample zero-shot
# a very small subset, for testing inference functionality
# This was model_wrapper_2, with metric computation and video saving disabled by config.
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=false \
#output_dir=outputs/2025-09-14_debug-test-002_dl3dv-zs-simt_depthsplat-Small-3DLPF-2DMip-Fullscratch \
#2>&1 | tee outputs/2025-09-14_debug-test-002_dl3dv-zs-simt_depthsplat-Small-3DLPF-2DMip-Fullscratch.log

# now testing: save upsampled rendering, skip upsampled scores
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=true \
#test.skip_upsampled_scores=true \
#output_dir=outputs/2025-09-14_debug-test-003_dl3dv-zs-simt_skip-upsampled-scores \
#2>&1 | tee outputs/2025-09-14_debug-test-003_dl3dv-zs-simt_skip-upsampled-scores.log

# now testing: skip upsampled rendering, skip upsampled scores
# file I/O for high-res results was indeed the bottleneck.
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#output_dir=outputs/2025-09-14_debug-test-004_dl3dv-zs-simt_skip-upsampled-scores-skip-upsampled-io \
#2>&1 | tee outputs/2025-09-14_debug-test-004_dl3dv-zs-simt_skip-upsampled-scores-skip-upsampled-io.log

############################################################
# RE10k 2x256x256 --> DL3DV 2x256x448 zeroshot full test set
############################################################

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Full mip-splatting + depthsplat (Small); 'second baseline'
# skip upsampled rendering, retain upsampled scores
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=false \
#output_dir=outputs/2025-09-14_test-005_dl3dv-256x448-2v-zs-simt_depthsplat-Small-3DLPF-2DMip-Fullscratch \
#2>&1 | tee outputs/2025-09-14_test-005_dl3dv-256x448-2v-zs-simt_depthsplat-Small-3DLPF-2DMip-Fullscratch.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# vanilla modelzoo depthsplat (Small); 'first baseline'
# skip upsampled rendering, retain upsampled scores
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=false \
#output_dir=outputs/2025-09-14_test-006_dl3dv-256x448-2v-zs-simt_depthsplat-Small-modelzoo \
#2>&1 | tee outputs/2025-09-14_test-006_dl3dv-256x448-2v-zs-simt_depthsplat-Small-modelzoo.log

############################################################
# RE10k 2x256x256 --> ACID 2x256x256 zeroshot full test set
############################################################

# RE10k 256x256 --> ACID 256x256 zero-shot full test set
# vanilla modelzoo depthsplat (Small); 'first baseline'
# skip upsampled rendering, retain upsampled scores
# skip image saving for now, just compute metrics quickly
#python -m src.main_2 +experiment=re10k \
#mode=test \
#dataset.roots=[datasets_extra/acid] \
#dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#test.save_image=false \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=false \
#output_dir=outputs/2025-09-14_test-008_acid-2x256x256-2v-zs-simt_depthsplat-Small-modelzoo \
#2>&1 | tee outputs/2025-09-14_test-008_acid-2x256x256-2v-zs-simt_depthsplat-Small-modelzoo.log

############################################################
# Debugging - trying out various test settings
############################################################

# now testing: what I think is saving the GS mean depths
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#output_dir=outputs/2025-09-22_debug-test-001_dl3dv-zs-simt_depth-visualization-settings \
#2>&1 | tee outputs/2025-09-22_debug-test-001_dl3dv-zs-simt_depth-visualization-settings.log

# Per-scene metric recording, in csv format
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#output_dir=outputs/2025-09-22_debug-test-002_dl3dv-zs-simt_per-scene-metrics \
#2>&1 | tee outputs/2025-09-22_debug-test-002_dl3dv-zs-simt_per-scene-metrics.log

# All-scenes metric recording, in csv format
# attempt 1 - csv does save, but the structure is all wrong
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#output_dir=outputs/2025-09-22_debug-test-003_dl3dv-zs-simt_comprehensive-metrics-table-csv \
#2>&1 | tee outputs/2025-09-22_debug-test-003_dl3dv-zs-simt_comprehensive-metrics-table-csv.log

# attempt 2 - worked that time
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#output_dir=outputs/2025-09-22_debug-test-004_dl3dv-zs-simt_comprehensive-metrics-table-csv \
#2>&1 | tee outputs/2025-09-22_debug-test-004_dl3dv-zs-simt_comprehensive-metrics-table-csv.log

# TODO: saving grid-based visualizations
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#output_dir=outputs/2025-09-22_debug-test-005_dl3dv-zs-simt_grid_vis_1x \
#2>&1 | tee outputs/2025-09-22_debug-test-005_dl3dv-zs-simt_grid_vis_1x.log

# rendering and saving gaussian depths in grid too
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#output_dir=outputs/2025-09-23_debug-test-003_dl3dv-zs-simt_grid_vis_renderdepth_1x \
#2>&1 | tee outputs/2025-09-23_debug-test-003_dl3dv-zs-simt_grid_vis_renderdepth_1x.log

# saving multires grid images (lower res only for now)
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#output_dir=outputs/2025-09-23_debug-test-006_dl3dv-zs-simt_grid_vis_downsampled \
#2>&1 | tee outputs/2025-09-23_debug-test-006_dl3dv-zs-simt_grid_vis_downsampled.log

# TODO: saving gaussians?
# saving multires grid images (lower res only for now)
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=100 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-23_debug-test-006_dl3dv-zs-simt_saving-ply \
#2>&1 | tee outputs/2025-09-23_debug-test-006_dl3dv-zs-simt_saving-ply.log

# TODO: confidence scores?

##########################################################################################
# Model Zoo checkpoints, detailed visuals & metrics
##########################################################################################

# Small Model, RE10K 2x256x256 vs RE10K 2x256x256
#python -m src.main_3 +experiment=re10k \
#mode=test \
#dataset.roots=[datasets/re10k] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset/view_sampler=evaluation \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-23_test-003_re10k-2x256x256-simt_depthsplat-modelzoo-Small_detailed-results \
#2>&1 | tee outputs/2025-09-23_test-003_re10k-2x256x256-simt_depthsplat-modelzoo-Small_detailed-results.log

# Small Model, RE10K 2x256x256 vs ACID 2x256x256
#python -m src.main_3 +experiment=re10k \
#mode=test \
#dataset.roots=[datasets_extra/acid] \
#dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset/view_sampler=evaluation \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-24_test-001_acid-zs-2x256x256-simt_depthsplat-modelzoo-Small_detailed-results \
#2>&1 | tee outputs/2025-09-24_test-001_acid-zs-2x256x256-simt_depthsplat-modelzoo-Small_detailed-results.log

# Small Model, RE10K 2x256x256 vs DL3DV 2x256x448
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset/view_sampler=evaluation \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-24_test-003_dl3dv-zs-2x256x448-simt_depthsplat-modelzoo-Small_detailed-results \
#2>&1 | tee outputs/2025-09-24_test-003_dl3dv-zs-2x256x448-simt_depthsplat-modelzoo-Small_detailed-results.log

# Base Model, RE10K 2x256x256 -> DL3DV {2,3,4,5,6}x256x448 vs DL3DV 2x256x448
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_2v_video_0_50.json \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-24_test-005_dl3dv-id-2x256x448-simt_depthsplat-modelzoo-Base_detailed-results \
#2>&1 | tee outputs/2025-09-24_test-005_dl3dv-id-2x256x448-simt_depthsplat-modelzoo-Base_detailed-results.log

## Base Model, RE10K 2x256x256 -> DL3DV {2,3,4,5,6}x256x448 vs DL3DV 4x256x448
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=4 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-24_test-006_dl3dv-id-4x256x448-simt_depthsplat-modelzoo-Base_detailed-results \
#2>&1 | tee outputs/2025-09-24_test-006_dl3dv-id-4x256x448-simt_depthsplat-modelzoo-Base_detailed-results.log

## Base Model, RE10K 2x256x256 -> DL3DV {2,3,4,5,6}x256x448 vs DL3DV 4x256x448
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.num_context_views=6 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=true \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.depth_mode=depth \
#test.save_grid_comparisons_downsampled=true \
#test.save_gaussian=true \
#output_dir=outputs/2025-09-25_test-001_dl3dv-id-6x256x448-simt_depthsplat-modelzoo-Base_detailed-results \
#2>&1 | tee outputs/2025-09-25_test-001_dl3dv-id-6x256x448-simt_depthsplat-modelzoo-Base_detailed-results.log

################################################################################
# RE10k 2x256x256 --> DL3DV 2x256x448 & 2x512x896 zeroshot full test set
################################################################################

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Full mip-splatting + depthsplat (Small); 'second baseline'
# skip upsampled rendering, retain upsampled scores
#python -m src.main_4 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=false \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.save_grid_comparisons_downsampled=true \
#test.save_grid_comparisons_upsampled=true \
#test.depth_mode=depth \
#output_dir=outputs/2025-09-29_test-001_dl3dv-2x256x448-2x512x896-zs-simt_depthsplat-Small-3DLPF-2DMip-Fullscratch \
#2>&1 | tee outputs/2025-09-29_test-001_dl3dv-2x256x448-2x512x896-zs-simt_depthsplat-Small-3DLPF-2DMip-Fullscratch.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Full mip-splatting + depthsplat (Small); 'second baseline'
# skip upsampled rendering, retain upsampled scores
# no model pass; just checking camera trajectories
#python -m src.main_4 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=false \
#test.save_depth=true \
#test.save_depth_concat_img=true \
#test.save_grid_comparisons=true \
#test.save_grid_comparisons_downsampled=true \
#test.save_grid_comparisons_upsampled=true \
#test.depth_mode=depth \
#output_dir=outputs/2025-09-29_test-002_dl3dv-2x256x448-2x512x896_camera-pose-check \
#2>&1 | tee outputs/2025-09-29_test-002_dl3dv-2x256x448-2x512x896_camera-pose-check.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Full mip-splatting + depthsplat (Small); 'second baseline'
# skip upsampled rendering, retain upsampled scores
# just one scene
python -m src.main_4 +experiment=dl3dv \
mode=test \
dataset.roots=[datasets_extra/dl3dv_480p] \
dataset/view_sampler=evaluation \
dataset.test_chunk_interval=1 \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
dataset.overfit_to_scene=06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_adapter.compensate_opacities=false \
model.decoder.name=splatting_cuda_anysplat \
model.decoder.rasterize_mode=antialiased \
model.decoder.eps2d=0.1 \
checkpointing.pretrained_model=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch/checkpoints/epoch_36-step_600000.ckpt \
test.save_video=false \
test.compute_scores=true \
test.save_image_upsampled=false \
test.skip_upsampled_scores=false \
test.save_depth=true \
test.save_depth_concat_img=true \
test.save_grid_comparisons=true \
test.save_grid_comparisons_downsampled=true \
test.save_grid_comparisons_upsampled=true \
test.depth_mode=depth \
output_dir=outputs/2025-09-29_test-003_dl3dv-2x256x448-2x512x896-zs-simt-subset_depthsplat-Small-3DLPF-2DMip-Fullscratch \
2>&1 | tee outputs/2025-09-29_test-003_dl3dv-2x256x448-2x512x896-zs-simt-subset_depthsplat-Small-3DLPF-2DMip-Fullscratch.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# vanilla modelzoo depthsplat (Small); 'first baseline'
# skip upsampled rendering, retain upsampled scores
#python -m src.main_3 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets_extra/dl3dv_480p] \
#dataset/view_sampler=evaluation \
#dataset.test_chunk_interval=1 \
#dataset.view_sampler.num_context_views=2 \
#dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#test.save_video=false \
#test.compute_scores=true \
#test.save_image_upsampled=false \
#test.skip_upsampled_scores=false \
#output_dir=outputs/2025-09-14_test-006_dl3dv-256x448-2v-zs-simt_depthsplat-Small-modelzoo \
#2>&1 | tee outputs/2025-09-14_test-006_dl3dv-256x448-2v-zs-simt_depthsplat-Small-modelzoo.log