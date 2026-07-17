# PromptMR+ pinned upstream subset

This directory preserves the exact PromptMR+ files used by the thin adapter from:

- Source: <https://github.com/hellopipu/PromptMR-plus>
- Commit: `934eeda6d4d18cd39e406fa1eee9e1f70603cb5e`
- License: Rutgers Non-commercial Research License (RU-NCRL), Rutgers docket `#2025-032`
- Workflow status: `NONCOMMERCIAL_COMPETITION_USE_ALLOWED`

`LICENSE.md` is the unmodified upstream license. Its copyright notice, conditions, and disclaimer must remain with source and binary redistributions. This integration does not grant or imply commercial-use rights.

`SOURCE_MANIFEST.json` records and verifies the exact license, source, config, and official checkpoint metadata. The files under `upstream/` and `configs/` are byte-identical to the pinned commit. Local behavior is limited to the adapter in `utils/model/promptmr_plus_adapter.py`; the upstream algorithm files are not edited.

Architecture feasibility, checkpoint namespace compatibility, competition quality, and official evaluation compatibility are separate gates. Prior competition use is not evidence for this year's score.
