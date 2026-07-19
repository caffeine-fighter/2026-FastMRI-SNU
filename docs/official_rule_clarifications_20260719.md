# Official rule clarifications — 2026-07-19

This document records the organizer answers that govern the current prize-first program.

## External or externally supplied initialization weights

[FastMRI challenge Issue #408](https://github.com/LISTatSNU/FastMRI_challenge/issues/408#issuecomment-5013509585) permits reproducible initialization functions and published initialization hyperparameters, but prohibits specific model weights obtained from external data or loaded from an external file.

Operational consequence: do not download, load, transcribe, fine-tune, ensemble, distill from, or submit the public PromptMR+ knee checkpoint. Architecture source and scalar recipes may be studied independently of those weights.

## External screening, augmentation, ensemble, and cloud compute

[FastMRI challenge Issue #409](https://github.com/LISTatSNU/FastMRI_challenge/issues/409#issuecomment-5013857396) confirms the following, provided final-submitted components are trained end-to-end on VESSL:

- LOCAL/RunPod may screen architectures and scalar hyperparameters using challenge-allowed data.
- Train/validation k-space masks and MRI coordinates may be transformed for learning, subject to non-redistribution and non-identification requirements.
- General ensemble and TTA are allowed under the stated no-leakage and timed-reconstruction conditions.
- Organizer train/validation data may be processed in a private external cloud while complying with the DSA.

## Implementation contract

- Transfer source code, architecture definitions, scalar hyperparameters, deterministic policies, and aggregate metrics only.
- Do not transfer learned model state, optimizer/scheduler/scaler state, EMA/SWA state, RNG state, teacher outputs, or reconstruction caches into a final VESSL run.
- Keep external cloud storage private; do not redistribute data or attempt patient/source identification.
- Apply augmentation/remasking only to organizer train/validation data and keep k-space, coils, adjacent slices, masks, targets, and coordinates physically consistent.
- Do not use target images, bbox annotations, labels, filenames, or leaderboard outcomes as inference inputs or routing signals.
- Run all submitted ensemble/TTA reconstruction computation inside the timed `recon_slice()` lifecycle.
- Preserve the official evaluator and mounted data bytes.

These permissions remove the former `BLOCKED_UNKNOWN_RULES` status. They do not remove model-specific license obligations or exact eligibility gates.
