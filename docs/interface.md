# Interface Contract

## Source of truth
- Clip index JSON: artifacts/clip_index_T8_s1_keyframes.json
- Each entry is a ClipSpec containing (scene_idx, start_sd_token, T, stride, keyframes_only)

## Clip loader invariants
- keyframes_only=True => all frames are sample_data keyframes
- No 'sweeps/' filenames
- Timestamps strictly increasing

## Later Dataset requirement
Given a ClipSpec, dataset must:
1) load T images from disk in order
2) apply a fixed resize/crop policy (to be decided later and then locked)
3) convert to a tensor with shape [T, C, H, W]
4) pixel range policy must be explicit (e.g., 0..1 float or -1..1 float)
5) return metadata (at least start_sd_token, filenames) for debugging
