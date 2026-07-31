# WPS DP Research

This directory is the artifact area for the KDocs/WPS browser automation under
[`wps_dp/`](..).

## Source entrypoints

- [`research.py`](../research.py): DrissionPage CLI for probe, copy,
  rename, and share flows.
- [`artifacts.py`](../artifacts.py): artifact naming and save-dir
  helpers.
- [`research.ipynb`](../research.ipynb): ad-hoc notebook experiments.

## Directory layout

- `runs/`: default output root for new CLI runs. Each run gets its own timestamped
  directory, for example `20260409_101530__copy-open__capdA7mscqov`.
- `_tmp/`: scratch screenshots and throwaway click experiments.
- root-level `*.json` / `*.png` and dated folders: historical outputs kept in place
  as archive data from the earlier flat layout.

## Standard files in one run

- `run_context.json`: request-level metadata for the run.
- `page.png` / `probe.json`: base page snapshot.
- `title_area.json`: title area inspection.
- `copy_result.json`: create-copy result, when `--create-copy` is used.
- `open_now_result.json`: opened-copy result, when `--open-now` is used.
- `rename_result.json`: rename result, when `--rename-title` is used.
- `share_result.json` / `share_settings.json`: share settings result and snapshot.
- `run_summary.json`: condensed summary of all actions in this run.

## Recommended commands

```powershell
uv run python -m wps_dp.research "https://www.kdocs.cn/l/xxxxx"
uv run python -m wps_dp.research "https://www.kdocs.cn/l/xxxxx" --create-copy --open-now
uv run python -m wps_dp.research "https://www.kdocs.cn/l/xxxxx" --rename-title "20260401第39届念住"
uv run python -m wps_dp.research "https://www.kdocs.cn/l/xxxxx" --share-enabled on --share-scope all --share-permission edit
```

Use `--label` when you want the auto-created run directory to carry a short business
hint, and `--save-dir` only when you need a fully fixed output path.
