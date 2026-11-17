# Intersection Monitor

Measures one signalized intersection with a Raspberry Pi (no video is ever
saved, only timestamped numbers) and recommends better signal timing on a laptop.

**Build status:** stage 1 of 4 is done (shared data format, sample data
generator, sample data). The collector, the analysis and the full student
guide come in stages 2-4; this file will be replaced by that guide.

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
