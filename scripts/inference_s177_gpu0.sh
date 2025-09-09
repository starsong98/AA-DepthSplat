export CUDA_VISIBLE_DEVICES=0
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
python -m src.main_2 +experiment=re10k \
dataset.test_chunk_interval=1 \
dataset.roots=[datasets/re10k] \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_adapter.compensate_opacities=false \
model.decoder.name=splatting_cuda_anysplat \
model.decoder.rasterize_mode=antialiased \
model.decoder.eps2d=0.1 \
checkpointing.pretrained_model=checkpoints/2025-09-08_train001_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullft/checkpoints/epoch_12-step_200000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
test.save_video=false \
output_dir=outputs/2025-09-09_test002_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft \
2>&1 | tee outputs/2025-09-09_test002_re10k-256x256-simt_depthsplat-Small-3DLPF-2DMip-Fullft.log