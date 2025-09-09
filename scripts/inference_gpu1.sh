export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

# 6 input views:
# Table 7 of depthsplat paper
# OOM for some reason
#python -m src.main +experiment=dl3dv \
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
#output_dir=outputs/2025-09-04_test001_dl3dv-256x448-6v-sist-fullset_baseline-Base \
#2>&1 | tee outputs/2025-09-04_test001_dl3dv-256x448-6v-sist-fullset_baseline-Base.log

python -m src.main_2 +experiment=dl3dv \
mode=test \
dataset.roots=[datasets_extra/dl3dv_480p] \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
test.save_video=false \
output_dir=outputs/2025-09-05_test001_dl3dv-256x448-6v-simt-fullset_baseline-Base \
2>&1 | tee outputs/2025-09-05_test001_dl3dv-256x448-6v-simt-fullset_baseline-Base.log

# 2 input views:
# Table 7 of depthsplat paper
#python -m src.main +experiment=dl3dv \
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
#output_dir=outputs/2025-09-04_test002_dl3dv-256x448-2v-sist-fullset_baseline-Base \
#2>&1 | tee outputs/2025-09-04_test002_dl3dv-256x448-2v-sist-fullset_baseline-Base.log