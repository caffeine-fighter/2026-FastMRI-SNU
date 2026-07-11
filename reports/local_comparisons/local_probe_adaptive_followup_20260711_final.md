# LOCAL adaptive final analysis

Generated: 2026-07-11T08:48:31+09:00

**Scope:** LOCAL desktop exploratory evidence only. This is not official Phase 2 evaluation or timing.

All five planned runs completed, each with 30 volumes, 791 slices, 161 bbox annotations, and zero skipped files.

## Matched comparisons

| Comparison | Full Δ | BBox Δ | Quality Δ | Result |
|---|---:|---:|---:|---|
| c4/ch16/s8 − c4/ch12/s8, seed 430, e5 | +0.002858331170 | +0.006849621394 | +0.004853976282 | candidate wins |
| c4/ch16/s8 − c4/ch12/s8, seed 431, e1 | -0.001529929988 | -0.000881058829 | -0.001205494408 | candidate loses |
| candidate e10 − candidate e5, seed 430 | +0.003468255930 | +0.004356303570 | +0.003912279750 | stable improvement |

The seed-430 one-epoch candidate gain was `+0.008021766148`, but the matched seed-431 gain reversed to `-0.001205494408`.

## Gate

- Matched e5 quality gain: **PASS** (`+0.004853976282`).
- Matched e5 full/bbox health: **PASS**.
- Seed-431 quality confirmation: **FAIL** (`-0.001205494408`).
- Seed-431 component health: **FAIL** (minimum component delta `-0.001529929988`, allowed `-0.001000000000`).
- Candidate e10 stability: **PASS** (`+0.003912279750` versus e5).
- Official VESSL resource/timing evidence: **PENDING**.

## Recommendation

**Do not promote c4/ch16/s8.** It lacks seed robustness under the predefined gate. EXP030 remains authoritative while EXP031 finishes and its file-backed VESSL handoff is validated.
