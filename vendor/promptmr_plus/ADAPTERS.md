# PromptMR+ local adapter boundary

Upstream architecture source under this directory is copied byte-for-byte from
`hellopipu/PromptMR-plus@934eeda6d4d18cd39e406fa1eee9e1f70603cb5e` and is
listed in `SOURCE_MANIFEST.json`.

The only local file inside the upstream namespace is:

- `data/__init__.py`: a deliberately empty package marker. It lets the unchanged
  upstream `models/utils.py` resolve `from data import transforms` after the runtime
  adapter places this pinned vendor root at the front of `sys.path`.

Repository-owned adapters live outside this vendor tree:

- `utils/promptmr/contracts.py`: immutable recipe, family, routing, and checkpoint metadata contracts.
- `utils/promptmr/data.py`: challenge HDF5 layout, five-slice boundary replication, and train-only 384 crop.
- `utils/promptmr/runtime.py`: model import/factory and center-slice output adapter.
- `utils/promptmr/planner.py`: metadata-only local run planning and resource estimates.
- `utils/learning/train_part.py` and `utils/learning/resume.py`: shared loop, optimizer/scheduler/AMP, and full-resume integration.

Do not edit the files listed in `SOURCE_MANIFEST.json`. Re-audit a new exact upstream
commit and regenerate every hash instead.
