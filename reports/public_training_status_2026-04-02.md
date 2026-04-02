# Public Training Status 2026-04-02

## Current Public Dataset Pipeline

The public clean-data training pipeline has been started on the dual-A800 server.

### Active job

- remote PID: `88472`
- remote log: `/root/lunwen_run/public_cremad_train.log`

### Current stage

The server is currently downloading the official CREMA-D mirror with `git-lfs`.

Started stages:

1. install `git-lfs`
2. clone `https://gitlab.com/cs-cooper-lab/crema-d-mirror.git`
3. pull large media files

Upcoming automatic stages:

4. build `datasets/public_cremad_trimodal`
5. train `public_cremad_trimodal_emotion_model.pt` on dual A800
6. write `reports/public_cremad_trimodal_emotion_metrics.json`

## Local Code Added

- `training/public_cremad.py`
- `scripts/build_cremad_trimodal_dataset.py`
- `scripts/server_start_public_cremad_training.sh`
- `scripts/server_start_public_pretrained_trimodal_care.sh`
- `training/configs/public_cremad_trimodal.yaml`

## Git Status

Pushed to GitHub in commit:

- `3bd68d6 feat: add public CREMA-D pretraining pipeline`

## Why This Matters

This is the first step toward replacing synthetic-only upstream perception with a clean public dataset pretraining stage.

Once the public emotion pretraining finishes, the next intended follow-up is:

1. use the public-pretrained encoder to initialize the tri-modal care model
2. compare against the synthetic-pretrained version
3. check whether weaklabel / downstream timing transfer improves
