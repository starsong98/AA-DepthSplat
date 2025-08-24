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
python -m src.main +experiment=dl3dv \
dataset.test_chunk_interval=1 \
dataset.roots=[datasets_extra/re10k_720p_test_subset] \
dataset.image_shape=[512,960] \
dataset.ori_image_shape=[720,1280] \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/re10k_ctx_6v_video.json \
test.save_video=true \
test.compute_scores=true \
test.render_chunk_size=10 \
output_dir=outputs/depthsplat-re10k-512x960_run012 2>&1 | tee outputs/20250821_run012_720p_subset.log