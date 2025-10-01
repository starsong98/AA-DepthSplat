export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

# full mipsplat + depthsplat finetuned prototype, sub test set, evaluation setting
#python -m src.main_2 +experiment=re10k \
#dataset.test_chunk_interval=1 \
#dataset.roots=[datasets/re10k_subset] \
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
#output_dir=outputs/2025-09-09_test002_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft \
#2>&1 | tee outputs/2025-09-09_test002_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Full mip-splatting + depthsplat (Small); 'second baseline'
# skip upsampled rendering, retain upsampled scores
# no model pass; just checking camera trajectories
#python -m src.main_4 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets/dl3dv_480p] \
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
#output_dir=outputs/2025-10-01_test-002_dl3dv-2x256x448-2x512x896_camera-pose-check \
#2>&1 | tee outputs/2025-10-01_test-002_dl3dv-2x256x448-2x512x896_camera-pose-check.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Naive modelzoo baseline
# skip all fancy stuff; just grab the numbers for now
#python -m src.main_4 +experiment=dl3dv \
#mode=test \
#dataset.roots=[datasets/dl3dv_480p] \
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
#test.save_depth=true \
#test.save_depth_concat_img=false \
#test.save_grid_comparisons=false \
#test.save_grid_comparisons_downsampled=false \
#test.save_grid_comparisons_upsampled=false \
#test.depth_mode=depth \
#output_dir=outputs/2025-10-01_test-004_dl3dv-2x256x448-2x512x896_numbers-only-modelzoo \
#2>&1 | tee outputs/2025-10-01_test-004_dl3dv-2x256x448-2x512x896_numbers-only-modelzoo.log

# RE10k 256x256 --> DL3DV 256x448 zero-shot full test set
# Full mip-splatting + depthsplat (Small); 'second baseline'
# skip upsampled rendering, retain upsampled scores
# no model pass; just checking camera trajectories
# NOW we can do all the fancy visualizations and stuff
python -m src.main_4 +experiment=dl3dv \
mode=test \
dataset.roots=[datasets/dl3dv_480p] \
dataset/view_sampler=evaluation \
dataset.test_chunk_interval=1 \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
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
output_dir=outputs/2025-10-01_test-006_dl3dv-2x256x448-2x512x896-zs-simt-ms_depthsplat-modelzoo-Small \
2>&1 | tee outputs/2025-10-01_test-006_dl3dv-2x256x448-2x512x896-zs-simt-ms_depthsplat-modelzoo-Small.log