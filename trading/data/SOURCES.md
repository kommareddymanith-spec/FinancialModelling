# Market data provenance

## `sp500_monthly.csv`

- **Series**: S&P 500 index level, monthly.
- **Rows**: 440 (1990-01-01 to 2026-08-01).
- **Retrieved**: 2026-09-10, from
  `https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv`
  (the `Date` and `SP500` columns; other columns dropped, rows before 1990
  trimmed). That dataset repackages Robert Shiller's long-run US equity series.
- **Important**: Shiller's `SP500` column is the **monthly average of daily
  closes**, not a month-end close. For a monthly savings plan this is a
  reasonable — arguably fairer — purchase price than a single day's close, but
  it is not a month-end mark, and drawdowns measured on it are month-average
  drawdowns that understate true intramonth falls.
- **Price return only.** The `Dividend` column is empty for recent months, so
  the benchmark computed from this file excludes dividends. A real S&P 500
  index fund reinvests them, so its total return runs roughly 1.2–1.5% a year
  above what this file shows. The benchmark understates the index fund it
  stands for, and any strategy compared against it is being flattered.

## What is missing

No historical **headline** archive is vendored here, because none was
obtainable: WSJ RSS feeds serve only the current ~30 items with no archive, and
`wsj.com`, `web.archive.org` and `api.gdeltproject.org` are all blocked by this
environment's egress policy. See the "Historical headline data" section of
`../README.md` for the sources that do carry it and how to load one.
