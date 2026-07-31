# WPS DP

This subdirectory groups the KDocs/WPS DrissionPage automation that had been
scattered at the repo root.

## Layout

- [`research.py`](D:\home\chenkunze\slns\kq5034\wps_dp\research.py): main CLI and browser automation flow.
- [`artifacts.py`](D:\home\chenkunze\slns\kq5034\wps_dp\artifacts.py): save-dir naming helpers.
- [`research.ipynb`](D:\home\chenkunze\slns\kq5034\wps_dp\research.ipynb): ad-hoc interactive notebook.
- [`tests/test_artifacts.py`](D:\home\chenkunze\slns\kq5034\wps_dp\tests\test_artifacts.py): focused tests for path generation.
- [`output/`](D:\home\chenkunze\slns\kq5034\wps_dp\output): screenshots, probes, copy/rename/share results, and historical experiment data.

## Recommended command

```powershell
uv run python -m wps_dp.research "https://www.kdocs.cn/l/xxxxx" --create-copy --open-now
```

Use `--label` to annotate an auto-created run directory. The default output root
is [`wps_dp/output/runs/`](D:\home\chenkunze\slns\kq5034\wps_dp\output\runs).
