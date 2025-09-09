# render video on re10k (need to have ffmpeg installed)
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

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
#test.render_chunk_size=5 \
#output_dir=outputs/depthsplat-re10k-512x960_run012 2>&1 | tee outputs/20250821_run012_720p_subset.log

# full official toy subset, single-scale inference single-scale rendering
# reference using original
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

# same settings, on separated out model wrapper
#python -m src.main_2 +experiment=re10k \
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
#output_dir=outputs/depthsplat-re10k_256x256_1_run001 2>&1 | tee outputs/20250824_run001_subset.log

# same checkpoint, same subset, but multiscale rendering (no multiscale metrics though yet)
#python -m src.main_2 +experiment=re10k \
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
#output_dir=outputs/depthsplat-re10k_256x256_1_run002 2>&1 | tee outputs/20250824_run002_simtrender_subset.log

# multi-resolution testing (1x & 1/2x), high-res subset
#python -m src.main_2 +experiment=dl3dv \
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
#output_dir=outputs/depthsplat-re10k-512x960_run005_simtrender_subset 2>&1 | tee outputs/20250824_run003_720p_simtrender_subset.log

# multi-resolution testing (1x & 1/2x & 1/4x), high-res subset
#python -m src.main_2 +experiment=dl3dv \
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
#output_dir=outputs/depthsplat-re10k-512x960_run006_simtrender_subset \
#2>&1 | tee outputs/20250824_run006_720p_simtrender_subset.log

# same checkpoint, same subset, but multiscale rendering, now includes 1/4 res. (no multiscale metrics though yet)
#python -m src.main_2 +experiment=re10k \
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
#output_dir=outputs/depthsplat-re10k_256x256_1_run007 2>&1 | tee outputs/20250824_run007_simtrender_subset.log

# same checkpoint, same subset, but multiscale rendering, now includes 2x & 4x res. (no multiscale metrics though yet)
#python -m src.main_2 +experiment=re10k \
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
#output_dir=outputs/depthsplat-re10k_256x256_simt_subset_run001 2>&1 | tee outputs/20250825_run001_simtrender_subset.log

# same checkpoint, same subset, evaluation setting (no multiscale metrics though yet)
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#output_dir=outputs/depthsplat-re10k_256x256_simt_subset_run002 2>&1 | tee outputs/20250825_run002_simtrender_subset.log

# same checkpoint, same subset, evaluation setting (no multiscale metrics though yet)
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#output_dir=outputs/depthsplat-re10k_256x256_simt_subset_run003 \
#2>&1 | tee outputs/20250825_run003_simtrender_subset.log

# same checkpoint, same subset, evaluation setting
# now saving 1/2 scale GT
#(no multiscale metrics though yet)
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-re10k_256x256_simt_subset_run004 \
#2>&1 | tee outputs/20250825_run004_simtrender_subset.log

# same checkpoint, same subset, evaluation setting
# now saving 1/4x ~ 4x scale GTs (actually 2x and 4x are not quite gts)
# and metrics on all GTs
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-re10k_256x256_simt_subset_run005 \
#2>&1 | tee outputs/20250825_run005_simtrender_subset.log

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

# same checkpoint, same subset, evaluation setting
# But now with 3D Gaussian LPF, NO FINETUNING
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-3DLPF-Large-re10k_256x256_simt_subset_run005 \
#2>&1 | tee outputs/20250901_run006_simtrender_subset_3DLPF.log

# 3D Gaussian LPF, NO FINETUNING
# full test set now
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-3DLPF-Large-re10k_256x256_simt_fullset_run001 \
#2>&1 | tee outputs/20250901_run001_simtrender_fullset_3DLPF.log

# RE10k 256x256, small checkpoint
# DepthSplat + 3D LPF, same rasterizer
# GS head only finetuned, 50k steps x batch size 8 x 2 GPUs
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_004/checkpoints/epoch_6-step_50000.ckpt \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-3DLPF-Small-re10k_256x256_GSHeadft_simt_subset_run001 \
#2>&1 | tee outputs/20250902_run001_simtrender_subset_3DLPF.log

# RE10k 256x256, small checkpoint
# DepthSplat baseline
# subset test set
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.name=depthsplat \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-Small-re10k_256x256_simt_subset_20250902_run002 \
#2>&1 | tee outputs/20250902_run002_simtrender_subset_DepthSplatSmall.log

# RE10k 256x256, small checkpoint
# DepthSplat + 3D LPF, w/o ft
# subset test set
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/depthsplat-Small-re10k_256x256_simt_subset_20250902_run003 \
#2>&1 | tee outputs/20250902_run002_simtrender_subset_DepthSplatSmall_3DLPF_noft.log

# RE10k 256x256, small checkpoint
# DepthSplat + 3D LPF, same rasterizer
# GS head only finetuned, 50k steps x batch size 8 x 2 GPUs
# full test set
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#checkpointing.pretrained_model=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_004/checkpoints/epoch_6-step_50000.ckpt \
#mode=test \
#dataset/view_sampler=evaluation \
#dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
#test.save_video=false \
#output_dir=outputs/20250902_run005_re10k-256x256-simt-fullset_3DLPF-GSHeadft-Small \
#2>&1 | tee outputs/20250902_run005_re10k-256x256-simt-fullset_3DLPF-GSHeadft-Small.log

# RE10k 256x256, small checkpoint
# DepthSplat + 3D LPF, same rasterizer
# GS head only finetuned, 100k steps x batch size 8 x 2 GPUs
# full test set
python -m src.main_2 +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_adapter.compensate_opacities=false \
checkpointing.pretrained_model=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_005/checkpoints/epoch_12-step_100000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
test.save_video=false \
output_dir=outputs/20250903_run001_re10k-256x256-simt-fullset_3DLPF-GSHeadft-Small \
2>&1 | tee outputs/20250903_run001_re10k-256x256-simt-fullset_3DLPF-GSHeadft-Small.log