#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=3e47c12c726946688f60a01031af30854ae44216

# 3D LPF + 2D AA via gsplat
# full model training from scratch
# same hyperparams as prev. run.
python -m src.main_2 +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
train.extended_visualization=true \
trainer.max_steps=300000 \
trainer.val_check_interval=0.25 \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_adapter.compensate_opacities=false \
model.decoder.name=splatting_cuda_anysplat \
model.decoder.rasterize_mode=antialiased \
model.decoder.eps2d=0.1 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
wandb.project=depthsplat_re10k \
output_dir=checkpoints/2025-09-09_train001_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch \
2>&1 | tee checkpoints/2025-09-09_train001_re10k-256x256_depthsplat-Small-3DLPF-2DMip-Fullscratch.log