#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=3
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
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#trainer.max_steps=100 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat-small_trials_002 \
#wandb.project=depthsplat_re10k \
#wandb.name=baseline-training-002 \
#2>&1 | tee checkpoints/20250828_train_trial_002

# now trying out validation configs
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#trainer.max_steps=100 \
#trainer.val_check_interval=100 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat-small_trials_004 \
#wandb.project=depthsplat_re10k \
#wandb.name=baseline-training-004 \
#2>&1 | tee checkpoints/20250829_train_trial_004.log

# slightly longer training - to see throughput better
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#trainer.max_steps=1000 \
#trainer.val_check_interval=1000 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat-small_trials_005 \
#wandb.project=depthsplat_re10k_debug \
#wandb.name=baseline-training-005 \
#2>&1 | tee checkpoints/20250829_train_trial_005.log

# what is the 'exaggerated' video?
# nice, I'm definitely sticking to this one
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#train.extended_visualization=true \
#trainer.max_steps=500 \
#trainer.val_check_interval=500 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat-small_trials_006 \
#wandb.project=depthsplat_re10k_debug \
#wandb.name=baseline-training-006 \
#2>&1 | tee checkpoints/20250829_train_trial_006.log

# trying out training progress bar now
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#train.extended_visualization=true \
#trainer.max_steps=500 \
#trainer.val_check_interval=500 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat-small_trials_007 \
#wandb.project=depthsplat_re10k_debug \
#wandb.name=baseline-training-007 \
#2>&1 | tee checkpoints/20250829_train_trial_007.log

# trying out finetuning now
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#train.extended_visualization=true \
#trainer.max_steps=10000 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat-small_trials_008 \
#wandb.project=depthsplat_re10k_debug \
#wandb.name=baseline-training-008 \
#2>&1 | tee checkpoints/20250829_train_trial_008.log

# try out the baseline full training, just to see how such a run is supposed to behave.
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#train.extended_visualization=true \
#trainer.max_steps=600000 \
#trainer.val_check_interval=0.25 \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
#checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat_small_baseline-001 \
#wandb.project=depthsplat_re10k \
#wandb.name=baseline-small-scratch-001_20250829 \
#2>&1 | tee checkpoints/20250829_baseline_small_full_001.log

# now trying out the GS head finetuning + LPF.
# JMP's hyperparam recommendations: 100k steps(compared to original 600k), same LR.
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
#python -m src.main +experiment=re10k \
#data_loader.train.batch_size=4 \
#dataset.test_chunk_interval=10 \
#train.extended_visualization=true \
#trainer.max_steps=100000 \
#trainer.val_check_interval=0.25 \
#optimizer.train_gs_head_only=true \
#model.encoder.name=depthsplat_lpf \
#model.encoder.upsample_factor=4 \
#model.encoder.lowest_feature_resolution=4 \
#checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
#output_dir=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_001 \
#wandb.project=depthsplat_re10k \
#wandb.name=GSLPF-small-GSHeadft-001_20250901 \
#2>&1 | tee checkpoints/20250901_GSLPF_GSHeadft_small_full_001.log

# training has been debugged - trying again.
# now trying out the GS head finetuning + LPF.
# JMP's hyperparam recommendations: 100k steps(compared to original 600k), same LR.
# 2x 4090s (24GB) for 600K steps, batch size 4 on each gpu
python -m src.main_2 +experiment=re10k \
data_loader.train.batch_size=4 \
dataset.test_chunk_interval=10 \
train.extended_visualization=true \
trainer.max_steps=100000 \
trainer.val_check_interval=0.25 \
optimizer.train_gs_head_only=true \
model.encoder.name=depthsplat_lpf \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
output_dir=checkpoints/re10k-256x256-depthsplat_small_GSLPF_GSHeadft_003 \
wandb.project=depthsplat_re10k \
wandb.name=GSLPF-small-GSHeadft-003_20250901 \
2>&1 | tee checkpoints/20250901_GSLPF_GSHeadft_small_full_003.log