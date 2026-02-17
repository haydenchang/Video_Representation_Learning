# Day 7 Probe Report

- Generated: `2026-02-17T08:08:38.154926+00:00`
- Decision: **PASS**

## Config Snapshot

- dataroot: `C:\DS\TPV\nuScenes`
- index_path: `artifacts\clip_index_T8_s1_keyframes.json`
- ckpt_path: `C:\DS\ssl-video-rep\artifacts\checkpoints\mvm_day8_improved_seed1_step6000.pt`
- pretrained_available: `True`
- pretrained_status: `available`
- seeds: `[1, 2, 3]`
- pooling_modes: `['mean_visible', 'mean_all', 'mean_excluding_cls']`
- optimizers: `['adamw', 'sgd']`
- epochs: `20`
- sanity_epochs: `80`
- only_sanity: `False`
- max_train_samples: `10000`
- max_val_samples: `3000`

## Sanity Checks

- all_passed: `True`
- scene_overlap_passed: `True`
- token_overlap_passed: `True`
- token_overlap_counts: `seed1:train=400 val=100 overlap=0, seed2:train=400 val=100 overlap=0, seed3:train=400 val=100 overlap=0`
- tiny_subset_overfit: `passed=True` `train_mAP=0.7075` `threshold=0.6000`
- shuffled_label: `passed=True` `ref_mAP=0.7421` `shuffled_mAP=0.5608` `drop=0.1812`
- shuffled_collapse_metric: `train_true_mAP`
- shuffled_val_mAP_check: `ref_val=0.5475` `shuffled_val=0.6885`
- shuffled_checksums: `train=35216d77b800bb8ca5749aba75df55b5` `shuffled=a720b8cad3b629dce5a5959eac664242` `first5_row_diff_count=5`
- first_batch_label_use: `used=60b8979100274a27f2e3aeef07c07bcb` `lookup=60b8979100274a27f2e3aeef07c07bcb`
- eval_target_checksums: `eval=a30945f3a0e7cb68ce4227a990141319` `val=a30945f3a0e7cb68ce4227a990141319` `train=35216d77b800bb8ca5749aba75df55b5`

## PASS Gate

- pooling: `mean_excluding_cls`
- optimizer: `adamw`
- pretrained mean±std mAP: `0.6015 ± 0.0465`
- random mean±std mAP: `0.5719 ± 0.0232`
- delta mean±std: `0.0296 ± 0.0241`
- required delta: `0.0100`

## Ablation Summary

| pooling | optimizer | pre_mean | rand_mean | delta_mean |
|---|---:|---:|---:|---:|
| mean_visible | adamw | 0.6071 | 0.6016 | 0.0055 |
| mean_visible | sgd | 0.5834 | 0.6234 | -0.0400 |
| mean_all | adamw | 0.5763 | 0.5947 | -0.0184 |
| mean_all | sgd | 0.5821 | 0.5905 | -0.0083 |
| mean_excluding_cls | adamw | 0.6015 | 0.5719 | 0.0296 |
| mean_excluding_cls | sgd | 0.6018 | 0.5834 | 0.0184 |

## Run Metrics

| pooling | optimizer | seed | pre_mAP | rand_mAP | delta_mAP | pre_bestF1 | rand_bestF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean_visible | adamw | 1 | 0.5857 | 0.5925 | -0.0068 | 0.6803 | 0.6803 |
| mean_visible | adamw | 2 | 0.5849 | 0.6121 | -0.0272 | 0.6827 | 0.6827 |
| mean_visible | adamw | 3 | 0.6508 | 0.6003 | 0.0504 | 0.6602 | 0.6602 |
| mean_visible | sgd | 1 | 0.5933 | 0.6278 | -0.0345 | 0.6803 | 0.6803 |
| mean_visible | sgd | 2 | 0.5641 | 0.6097 | -0.0457 | 0.6827 | 0.6827 |
| mean_visible | sgd | 3 | 0.5929 | 0.6326 | -0.0397 | 0.6602 | 0.6602 |
| mean_all | adamw | 1 | 0.5611 | 0.6057 | -0.0446 | 0.6803 | 0.6803 |
| mean_all | adamw | 2 | 0.5545 | 0.5702 | -0.0156 | 0.6827 | 0.6827 |
| mean_all | adamw | 3 | 0.6132 | 0.6082 | 0.0050 | 0.6602 | 0.6602 |
| mean_all | sgd | 1 | 0.5670 | 0.5797 | -0.0128 | 0.6803 | 0.6803 |
| mean_all | sgd | 2 | 0.5645 | 0.5708 | -0.0063 | 0.6827 | 0.6827 |
| mean_all | sgd | 3 | 0.6150 | 0.6209 | -0.0059 | 0.6602 | 0.6602 |
| mean_excluding_cls | adamw | 1 | 0.5587 | 0.5462 | 0.0125 | 0.6803 | 0.6803 |
| mean_excluding_cls | adamw | 2 | 0.5797 | 0.5673 | 0.0125 | 0.6827 | 0.6827 |
| mean_excluding_cls | adamw | 3 | 0.6661 | 0.6024 | 0.0637 | 0.6602 | 0.6602 |
| mean_excluding_cls | sgd | 1 | 0.5309 | 0.5392 | -0.0082 | 0.6803 | 0.6803 |
| mean_excluding_cls | sgd | 2 | 0.6517 | 0.5747 | 0.0770 | 0.6827 | 0.6827 |
| mean_excluding_cls | sgd | 3 | 0.6228 | 0.6362 | -0.0134 | 0.6602 | 0.6602 |

## Alignment Samples

| clip_index | start_sd_token | sample_token | timestamp | num_positive_labels |
|---:|---|---|---:|---:|
| 113 | d41cafde14bf43ed8a5c994c09b02d0e | 684acda1d78c4a959762df565cba29f5 | 1531885321412463 | 2 |
| 334 | bb86f325591641c182aebf75e9e732cc | 03f54952344b4fd29253e26557b6d56f | 1531886384612469 | 2 |
| 498 | 131b7d1ff24c45bc802b9c336b57e807 | 1b63a37d82ed4debbfdad9791a693d84 | 1532402330112460 | 3 |
| 300 | 0d81f6d885fa480b8b6d1655200561f1 | 64dd47bcaabc44279c9be12aab064752 | 1531886222512465 | 3 |
| 56 | 6d93bd8847424bb2a72e22eba23fc2c0 | 81a0b7e1553a42a7a77ecfbb423df537 | 1531884159912464 | 4 |
| 176 | 99d734fdca904cab9e1cf0d9e86006f3 | 2ff5689a089844f0a2097b1883916003 | 1531885795012464 | 4 |
| 246 | 56b8a282dab54cdda962b03d25e7201f | 8a2f1d422b5344bf856ac21c65f2d562 | 1531886072512465 | 2 |
| 38 | 199343168de6408c8318957d159fe41b | 96253fd0fbb140268a12a15c98cc8e34 | 1531883989862460 | 5 |
| 265 | af6dcbee4b1443a980b8682e02774689 | 01df0413f9bb47478af656d87047fd6a | 1531886120912462 | 3 |
| 168 | 86285576f3ca4e70a0f65aaac5102a1b | 40847c1f1fd34cf2bdf278782dba841a | 1531885768912470 | 3 |

## Decision Reasoning

- Pretrained exceeds random by >= 0.01 mAP on gate config across seeds [1,2,3].
- Gate delta mean±std = 0.0296 ± 0.0241, pretrained mAP mean±std = 0.6015 ± 0.0465, random mAP mean±std = 0.5719 ± 0.0232.
