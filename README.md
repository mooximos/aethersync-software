# Intersection Monitor

Measures one signalized intersection with a Raspberry Pi (no video is ever
saved, only timestamped numbers) and recommends better signal timing on a laptop.

**Build status:** stages 1-3 of 4 are done (shared data format, sample data
generator, sample data, the Raspberry Pi collector and the laptop analysis). The
full student guide comes in stage 4; this file will be replaced by it.
See docs/BUILD_NOTES.md for the defaults chosen so far.

## What works now

```bash
pip install -r requirements-analysis.txt
```

```bash
python -m pytest
```

```bash
python tools/make_sample_data.py --out my_sample --days 1 --hours 7
```

Run the whole analysis on the sample data (writes results/REPORT.html):

```bash
PYTHONPATH=.:analysis python -m tlopt all sample_data --intersection analysis/intersection.example.yaml
```

- `common/` - the data format both programs share (column lists, validators, time helpers, file readers)
- `tools/make_sample_data.py` - makes synthetic data in the exact format
- `sample_data/` - 2 weekdays x hours 07 and 17, made by the generator with its default settings
- `collector/` - the Pi program (`python -m monitor`), its installer and `check_hardware.py`
- `analysis/` - the laptop program (`python -m tlopt`): measures, HCM model, timing optimizer, SUMO check, report
