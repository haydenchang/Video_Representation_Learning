# Day 0-6 Correctness Audit Report

- Generated: `2026-02-17T07:35:14.831052+00:00`
- Decision: **PASS**

## Environment

- git_commit: `0a7df36d55f7c61838b67a2eceafaea74f55b4ec`
- python_version: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- torch_version: `2.9.1+cpu`
- device: `cpu`
- dataroot: `C:\DS\TPV\nuScenes`
- index_path: `artifacts\clip_index_T8_s1_keyframes.json`

## PASS/FAIL Table

| Section | Status | Reason | Evidence |
|---|---|---|---|
| A. nuScenes init & keyframes-only invariants | PASS | Keyframe-only CAM_FRONT walker returns only keyframes and no sweeps paths. | sampled_count=20, sweeps_violations=0 |
| B. Clip index integrity | PASS | Sampled clip index entries satisfy length/timestamp/keyframe/existence invariants. | sampled_entries=50, bad_count=0 |
| C. Dataset tensor correctness (ClipDataset) | PASS | ClipDataset item[0] tensor contract holds and visualization artifact was produced. | shape=(8, 3, 360, 640), dtype=torch.float32, range=[0.0157,1.0000] |
| D. tubeletify / untubeletify round-trip | PASS | tubeletify/untubeletify round-trip is exact with expected token shape. | token_shape=(2, 576, 9600), max_err=0.0 |
| E. Mask generation determinism + masked loss correctness | PASS | Masking RNG is deterministic on CPU with fixed seed and masked loss unit checks pass. | mask_deterministic=True, loss0=0.0, loss1=1.000000 |
| F. Model forward/backward stability (MaskedVideoModel) | PASS | Five deterministic CPU train steps run with finite stats and no transformer warnings. | step0_loss=0.220019, step4_loss=0.212169, transformer_warning_count=0 |
| G. Day 6 instrumentation consistency | PASS | Activation recorder captured all required modules and latent feature spread is finite/non-collapsed. | activation_keys_ok=True, latent_feat_std=0.925043 |

## Artifacts

- report_md: `artifacts\audit_day0_6_report.md`
- report_json: `artifacts\audit_day0_6_report.json`
- dataset_viz_png: `artifacts\audit_viz_dataset_item.png`

## Section Evidence

### A. nuScenes init & keyframes-only invariants

- Status: **PASS**
- Reason: `Keyframe-only CAM_FRONT walker returns only keyframes and no sweeps paths.`

```json
{
  "scene_idx": 0,
  "start_sd_token": "020d7b4f858147558106c504f7f31bef",
  "sampled_count": 20,
  "example_filenames_and_timestamps": [
    {
      "sd_token": "020d7b4f858147558106c504f7f31bef",
      "filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg",
      "timestamp": 1531883530412470
    },
    {
      "sd_token": "27460c51459c46a6b8a94525793ff813",
      "filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530912460.jpg",
      "timestamp": 1531883530912460
    },
    {
      "sd_token": "6ff2a727cdd447c5956582f420c5a80c",
      "filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883531412477.jpg",
      "timestamp": 1531883531412477
    },
    {
      "sd_token": "5af8e4f03044406a9abd5bffd940952a",
      "filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883531912467.jpg",
      "timestamp": 1531883531912467
    },
    {
      "sd_token": "8d9b8b382be84198841f9409d7d38557",
      "filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883532412464.jpg",
      "timestamp": 1531883532412464
    }
  ]
}
```

### B. Clip index integrity

- Status: **PASS**
- Reason: `Sampled clip index entries satisfy length/timestamp/keyframe/existence invariants.`

```json
{
  "index_path": "artifacts\\clip_index_T8_s1_keyframes.json",
  "index_size": 500,
  "sampled_entries": 50,
  "sampled_indices": [
    1,
    4,
    11,
    13,
    14,
    15,
    32,
    48,
    52,
    60
  ],
  "bad_count": 0
}
```

### C. Dataset tensor correctness (ClipDataset)

- Status: **PASS**
- Reason: `ClipDataset item[0] tensor contract holds and visualization artifact was produced.`

```json
{
  "dataset_len": 500,
  "shape": [
    8,
    3,
    360,
    640
  ],
  "dtype": "torch.float32",
  "min": 0.01568627543747425,
  "max": 1.0,
  "start_token": "020d7b4f858147558106c504f7f31bef",
  "first_filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg",
  "last_filename": "samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883533912468.jpg",
  "viz_path": "artifacts\\audit_viz_dataset_item.png"
}
```

### D. tubeletify / untubeletify round-trip

- Status: **PASS**
- Reason: `tubeletify/untubeletify round-trip is exact with expected token shape.`

```json
{
  "token_shape": [
    2,
    576,
    9600
  ],
  "max_err": 0.0
}
```

### E. Mask generation determinism + masked loss correctness

- Status: **PASS**
- Reason: `Masking RNG is deterministic on CPU with fixed seed and masked loss unit checks pass.`

```json
{
  "mask_equal_with_seeded_cpu_generators": true,
  "loss0": 0.0,
  "loss1": 1.0
}
```

### F. Model forward/backward stability (MaskedVideoModel)

- Status: **PASS**
- Reason: `Five deterministic CPU train steps run with finite stats and no transformer warnings.`

```json
{
  "device": "cpu",
  "steps_ran": 5,
  "step_0": {
    "loss": 0.22001883387565613,
    "grad_norm": 0.01052609713911452,
    "param_norm": 393.0474566343714,
    "pred_mean": -0.00039817579090595245,
    "pred_std": 0.06673641502857208,
    "pred_absmax": 0.9691773653030396,
    "latent_feat_std": 0.9192376732826233,
    "activation_stats": {
      "in_proj": {
        "mean": -1.5917523342068307e-05,
        "std": 0.252521276473999,
        "absmax": 1.33589768409729
      },
      "encoder": {
        "mean": -4.11066514161007e-09,
        "std": 1.00002920627594,
        "absmax": 4.042568206787109
      },
      "out_proj": {
        "mean": -0.00039817579090595245,
        "std": 0.06673641502857208,
        "absmax": 0.9691773653030396
      }
    }
  },
  "step_4": {
    "loss": 0.21216930449008942,
    "grad_norm": 0.012243429852758086,
    "param_norm": 393.04256455041343,
    "pred_mean": 0.007155219092965126,
    "pred_std": 0.0666767880320549,
    "pred_absmax": 0.9615463614463806,
    "latent_feat_std": 0.9250432848930359,
    "activation_stats": {
      "in_proj": {
        "mean": 0.0003561737248674035,
        "std": 0.25909000635147095,
        "absmax": 1.3950293064117432
      },
      "encoder": {
        "mean": 1.0276662854025176e-09,
        "std": 1.0000091791152954,
        "absmax": 4.1976318359375
      },
      "out_proj": {
        "mean": 0.007155219092965126,
        "std": 0.0666767880320549,
        "absmax": 0.9615463614463806
      }
    }
  },
  "transformer_warning_count": 0,
  "all_warning_count": 0
}
```

### G. Day 6 instrumentation consistency

- Status: **PASS**
- Reason: `Activation recorder captured all required modules and latent feature spread is finite/non-collapsed.`

```json
{
  "activation_keys_present": [
    "in_proj",
    "encoder",
    "out_proj"
  ],
  "activation_stats_step4": {
    "in_proj": {
      "mean": 0.0003561737248674035,
      "std": 0.25909000635147095,
      "absmax": 1.3950293064117432
    },
    "encoder": {
      "mean": 1.0276662854025176e-09,
      "std": 1.0000091791152954,
      "absmax": 4.1976318359375
    },
    "out_proj": {
      "mean": 0.007155219092965126,
      "std": 0.0666767880320549,
      "absmax": 0.9615463614463806
    }
  },
  "latent_feat_std_step4": 0.9250432848930359
}
```
