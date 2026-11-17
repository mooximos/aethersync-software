# Intersection Monitor

Measures one signalized intersection with a Raspberry Pi (no video is ever
saved, only timestamped numbers) and recommends better signal timing on a laptop.

**Build status:** stages 1 and 2 of 4 are done (shared data format, sample data
generator, sample data, and the Raspberry Pi collector). The analysis and the full
student guide come in stages 3-4; this file will be replaced by that guide.
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

- `common/` - the data format both programs share (column lists, validators, time helpers, file readers)
- `tools/make_sample_data.py` - makes synthetic data in the exact format
- `sample_data/` - 2 weekdays x hours 07 and 17, made by the generator with its default settings
- `collector/` - the Pi program (`python -m monitor`), its installer and `check_hardware.py`
