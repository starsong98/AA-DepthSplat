#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=3e47c12c726946688f60a01031af30854ae44216

### REFERENCE FROM AUTHORS - BASE MODEL DL3DV 2x256x448 ~ 6x256x448
# finetune on dl3dv, random view 2-6
# train on 8x GPUs (>=80GB VRAM) for 100K steps, batch size 1 on each gpu
# resume from the previously pretrained model on re10k
#python -m src.main +experiment=dl3dv \
#data_loader.train.batch_size=1 \
#dataset.roots=[datasets/dl3dv] \
#dataset.view_sampler.num_target_views=8 \
#dataset.view_sampler.num_context_views=6 \
#dataset.min_views=2 \
#dataset.max_views=6 \
#trainer.max_steps=100000 \
#trainer.num_nodes=2 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-76a0605a.pth \
#wandb.project=depthsplat \
#output_dir=checkpoints/dl3dv-256x448-depthsplat-base-randview2-6

### REFERENCE FROM ME - SMALL MODEL RE10K 2x256x256
# 3D LPF + 2D AA via gsplat
# full model training from scratch
# same hyperparams as prev. run.
#python -m src.main_2 +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#train.extended_visualization=true \
#trainer.max_steps=600000 \
#trainer.val_check_interval=0.25 \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#wandb.project=depthsplat_re10k \
#output_dir=checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch \
#2>&1 | tee checkpoints/2025-09-09_train002_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch.log

# Now trying out myself
# 2x 4090s (24GB) for 400K steps, batch size 1 on each gpu
# rewired the checkpoint path because the checkpoint in the reference script does not exist
# also single node training too
# always 6 views; to see if OOM occurs.
#python -m src.main +experiment=dl3dv \
#data_loader.train.batch_size=1 \
#dataset.roots=[datasets/dl3dv_480p] \
#dataset.view_sampler.num_target_views=8 \
#dataset.view_sampler.num_context_views=6 \
#dataset.min_views=6 \
#dataset.max_views=6 \
#trainer.max_steps=100 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.name=depthsplat_lpf \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth \
#wandb.project=depthsplat_dl3dv_debug \
#output_dir=checkpoints/2025-09-16_debug-train-001_dl3dv-256x448-depthsplat-base-randview6-6_3DLPF-2DMip-Fulltrain \
#2>&1 | tee checkpoints/2025-09-16_debug-train-001_dl3dv-256x448-depthsplat-base-randview6-6_3DLPF-2DMip-Fulltrain.log

# Now trying out myself
# 2x 4090s (24GB) for 400K steps, batch size 1 on each gpu
# rewired the checkpoint path because the checkpoint in the reference script does not exist
# also single node training too
#python -m src.main_3 +experiment=dl3dv \
#data_loader.train.batch_size=1 \
#dataset.roots=[datasets/dl3dv_480p] \
#dataset.view_sampler.num_target_views=8 \
#dataset.view_sampler.num_context_views=6 \
#dataset.min_views=2 \
#dataset.max_views=6 \
#trainer.max_steps=400000 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.name=depthsplat_lpf \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth \
#wandb.project=depthsplat_dl3dv \
#output_dir=checkpoints/2025-09-16_train-006_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain \
#2>&1 | tee checkpoints/2025-09-16_train-006_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain.log

# attempting to resume...
#python -m src.main_3 +experiment=dl3dv \
#data_loader.train.batch_size=1 \
#dataset.roots=[datasets/dl3dv_480p] \
#dataset.view_sampler.num_target_views=8 \
#dataset.view_sampler.num_context_views=6 \
#dataset.min_views=2 \
#dataset.max_views=6 \
#trainer.max_steps=400000 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.name=depthsplat_lpf \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth \
#checkpointing.resume=true \
#wandb.id=z4x146zu \
#wandb.project=depthsplat_dl3dv \
#output_dir=checkpoints/2025-09-16_train-006_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain \
#2>&1 | tee checkpoints/2025-09-16_train-006-resume-001_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain.log

# Got much further than last time. attempting to resume again.
#python -m src.main_3 +experiment=dl3dv \
#data_loader.train.batch_size=1 \
#dataset.roots=[datasets/dl3dv_480p] \
#dataset.view_sampler.num_target_views=8 \
#dataset.view_sampler.num_context_views=6 \
#dataset.min_views=2 \
#dataset.max_views=6 \
#trainer.max_steps=400000 \
#model.encoder.num_scales=2 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=8 \
#model.encoder.monodepth_vit_type=vitb \
#model.encoder.name=depthsplat_lpf \
#model.encoder.gaussian_adapter.compensate_opacities=false \
#model.decoder.name=splatting_cuda_anysplat \
#model.decoder.rasterize_mode=antialiased \
#model.decoder.eps2d=0.1 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth \
#checkpointing.resume=true \
#wandb.id=z4x146zu \
#wandb.project=depthsplat_dl3dv \
#output_dir=checkpoints/2025-09-16_train-006_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain \
#2>&1 | tee checkpoints/2025-09-16_train-006-resume-002_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain.log

# Maybe this time...?
python -m src.main_3 +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.roots=[datasets/dl3dv_480p] \
dataset.view_sampler.num_target_views=8 \
dataset.view_sampler.num_context_views=6 \
dataset.min_views=2 \
dataset.max_views=6 \
trainer.max_steps=400000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.name=depthsplat_lpf \
model.encoder.gaussian_adapter.compensate_opacities=false \
model.decoder.name=splatting_cuda_anysplat \
model.decoder.rasterize_mode=antialiased \
model.decoder.eps2d=0.1 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x448-view2-fea94f65.pth \
checkpointing.resume=true \
wandb.id=z4x146zu \
wandb.project=depthsplat_dl3dv \
output_dir=checkpoints/2025-09-16_train-006_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain \
2>&1 | tee checkpoints/2025-09-16_train-006-resume-003_dl3dv-256x448-depthsplat-base-randview2-6_3DLPF-2DMip-Fulltrain.log