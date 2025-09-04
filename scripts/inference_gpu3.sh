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
python -m src.main_2 +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[datasets/re10k_subset] \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_adapter.compensate_opacities=false \
checkpointing.pretrained_model=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_005/checkpoints/epoch_12-step_100000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
test.save_video=false \
output_dir=outputs/2025-09-03_run002_re10k-256x256-simt-subset_3DLPF-GSHeadft-Small \
2>&1 | tee outputs/2025-09-03_run002_re10k-256x256-simt-subset_3DLPF-GSHeadft-Small.log