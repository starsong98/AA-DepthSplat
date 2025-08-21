# render video on re10k (need to have ffmpeg installed)
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

# render video on re10k (need to have ffmpeg installed)
# 2 input views at 256x256 resolutions
python -m src.main +experiment=re10k \
dataset.test_chunk_interval=100 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.compute_scores=false
output_dir=outputs/depthsplat-re10k

# let's try subset
# 2 input views at 256x256 resolutions
#python -m src.main +experiment=re10k_subset \
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
#test.compute_scores=false
#output_dir=outputs/depthsplat-re10k-subset