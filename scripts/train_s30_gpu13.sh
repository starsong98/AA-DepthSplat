#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=3e47c12c726946688f60a01031af30854ae44216


# Full model finetuning
python -m src.main_2 +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
train.extended_visualization=true \
trainer.max_steps=50000 \
trainer.val_check_interval=0.25 \
optimizer.train_gs_head_only=false \
optimizer.lr_gshead=2e-4 \
optimizer.lr=2e-5 \
optimizer.lr_monodepth=1e-6 \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_adapter.compensate_opacities=false \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
wandb.project=depthsplat_re10k \
output_dir=checkpoints/2025-09-05_train001_re10k-256x256_depthsplat-Small-3DLPF-Fullft \
2>&1 | tee checkpoints/2025-09-05_train001_re10k-256x256_depthsplat-Small-3DLPF-Fullft.log