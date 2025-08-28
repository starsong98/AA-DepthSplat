#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=3e47c12c726946688f60a01031af30854ae44216

### REFERENCE FROM AUTHORS - SMALL MODEL
# a single A100 (80GB) for 600K steps, batch size 8 on each gpu
# python -m src.main +experiment=re10k \
# data_loader.train.batch_size=8 \
# dataset.test_chunk_interval=10 \
# trainer.max_steps=600000 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=4 \
# checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
# checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
# output_dir=checkpoints/re10k-256x256-depthsplat-small

# Now trying out myself
# a single A100 (80GB) for 600K steps, batch size 8 on each gpu
python -m src.main +experiment=re10k \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=600000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
output_dir=checkpoints/re10k-256x256-depthsplat-small_trials \

