# recreating the debugger command that worked
export CUDA_VISIBLE_DEVICES=0
#python -m src.main +experiment=re10k \
#dataset.test_chunk_interval=1 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=2 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.monodepth_vit_type=vitl \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
#mode=test \
#dataset/view_sampler=evaluation

# Large model evaluation + logging
python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
mode=test \
dataset/view_sampler=evaluation \
output_dir=outputs/depthsplat-re10k-large \
2>&1 | tee outputs/logging_trials_002.log