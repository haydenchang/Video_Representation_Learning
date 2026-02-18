# Self-Supervised Video Representation Learning (nuScenes CAM_FRONT)

## Objective (Fixed)
Masked Video Modeling (VideoMAE-style): reconstruct masked spatiotemporal patches from visible context.

## Dataset (Fixed)
nuScenes v1.0
Sensor: CAM_FRONT only
No multi-camera fusion.
No LiDAR, radar, or maps for training.

## Backbone (Fixed for v1)
CNN patch embed + temporal transformer encoder + lightweight reconstruction head

## Clip Spec (Initial Fixed)
T = 6–8 frames
Temporal stride: configurable, but default stride = 1
Resolution: 360x640 

## What this project is NOT doing (Fixed)
- No BEV occupancy, no 3D occupancy
- No planning, no control
- No end-to-end autonomy stack
- No SOTA chasing

## Gating Checkpoints (Fixed)
- Stable training locally (loss decreases, no NaNs, no collapse signals)
- Linear probe beats random baseline before any scaling
- Dataset scaling curve produced on RCAC
- DDP throughput analysis produced on RCAC
- Temporal tradeoff analysis produced on RCAC

## Stop Conditions (Fixed)
If unstable > 24 hrs: simplify model / reduce resolution / reduce T / disable fancy components.
If probe does not beat random: fix representation learning before scaling.
