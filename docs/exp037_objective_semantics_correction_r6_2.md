# EXP037 objective semantics correction R6.2

This is an additive correction. It does not modify or replace immutable EXP037 training, validation, terminal-gate, or decision artifacts.

## Correct immutable objective

The actual Candidate objective was:

```text
(1 - foreground SSIM)
+ 1.0309823159913825 × normalized L1
```

Therefore, descriptions of EXP037 as either a normalized-L1-free objective or an EXP034/ScoreAlignedLoss replication are superseded.

This negative result is limited to the exact c8/ch12/s8 architecture, coefficient `1.0309823159913825`, frozen recipe digest `f51cac9e89ba515e852055ce241c4458a158cc07a906749bc1bd53a915b7322d`, seed 430, and matched continuation. It is not evidence that the normalized-L1 family as a whole fails.

## Frozen terminal result

- Terminal gate: `PASS`
- Candidate win fraction: `4.66%`
- Epoch-35 paired CI: `[-0.0002093725, -0.0000315047]`
- Best-quality delta versus protected EXP035: `+0.0000755733`
- acc8 full delta: `-0.0000027963`
- Four protected-cell non-regression: `FAIL`
- Final decision: `EXP037_RESEARCH_ONLY_REJECT`
- Official evaluation authorized: `false`
- Submission authorized: `false`

EXP037 is permanently excluded from official evaluation and submission.

## Objectives after EXP037

- Primary: eligible final rank <= 5 and prize
- Stretch: rank 1
- `target_score=null`
- Fixed scores such as 0.94 are not launch, promotion, failure, freeze, or stop gates.
- The historical `+0.0005` threshold applies only to the preregistered EXP037 matched branch, not to the prize objective.

## Evidence

- Correction JSON SHA-256: `a4ae15feed8fa10be8a5923fa1a8169496f9de04b3519cb4682da0e35d13254f`
- Independent review: `PASS` (`deleg_617018f9`)
- Review receipt SHA-256: `58c3c76a9ef37d6037d97b3d12ff7649e5c0ef4a58bf248671e986687dc9adb1`
- Terminal-gate SHA-256: `da10565df5307367a28ae5af7f712264482bf3d2b85d86b8c888065885b4140b`
- Reconciliation SHA-256: `d8d44129b266cc885b2ff3076ed1462cd4e69810139be4e7e8ef22535eb2ee6f`

See the additive JSON artifacts in `reports/phase2/` for all exact source, recipe, validation, bootstrap, and decision bindings.
