# Data dictionary

Machine-readable version: [`data/datapackage.json`](data/datapackage.json) (a [Frictionless Data Package](https://specs.frictionlessdata.io/data-package/), validated in CI together with file sizes and SHA-256 checksums).

All files are **unmodified third-party copies** held in `data/raw/`. The code in this repository is MIT-licensed; the datasets remain the property of their original owners and no licence was stated by them. They are included for research and education. Files derived by the code are written to `data/processed/`, which is not tracked.

## Provenance

| Dataset | Origin | How it reached this repository |
|---|---|---|
| Cookie Cats A/B test | Released for DataCamp's "Mobile Games A/B Testing with Cookie Cats" project (game by Tactile Entertainment) | Copy taken from a public GitHub mirror |
| CDNOW | Made available by Peter Fader and Bruce Hardie for research use | Distributed with the open-source [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) package, from which these files were taken |

## `data/raw/cookie_cats.csv`

One row per player; 90,189 rows; `userid` is unique. Used by notebook 1 and `sql/experiment_metrics.sql`.

| Field | Type | Meaning |
|---|---|---|
| `userid` | integer | Unique player identifier |
| `version` | string | First progress gate assigned at random on install: `gate_30` (control, 44,700 players) or `gate_40` (treatment, 45,489 players) |
| `sum_gamerounds` | integer | Game rounds played during the first 14 days after install (per the original DataCamp description); range 0 to 49,854 |
| `retention_1` | boolean | `TRUE` if the player came back and played 1 day after installing |
| `retention_7` | boolean | `TRUE` if the player came back and played 7 days after installing |

Quality notes: 3,994 players have 0 rounds; one account has 49,854 rounds (an outlier, handled with robust engagement metrics in the notebook); the split between arms is 50.44% / 49.56%, a slight imbalance that the analysis stress-tests.

## `data/raw/CDNOW_master.txt`

Whitespace-aligned text with a header row; 69,659 purchases by 23,570 customers, 1997-01-01 to 1998-06-30. Read by `prodanalytics.data.load_cdnow_transactions()`.

| Field | Type | Meaning |
|---|---|---|
| `customer_id` | string | Customer identifier, zero-padded to five digits (kept as text) |
| `date` | string | Purchase date as `YYYYMMDD` |
| `number_of_cds` | integer | Units bought in the purchase (1 to 99; renamed `units` by the loader) |
| `dollar_value` | number | Value of the purchase in US dollars (0.00 to 1,286.01; 80 purchases have value 0; renamed `value` by the loader) |

## `data/raw/CDNOW_sample.txt`

Whitespace-aligned text with **no header row**; 6,919 purchases by a 2,357-customer sample. Columns, in order:

| Position | Meaning |
|---|---|
| 1 | `customer_id`, as in the master file |
| 2 | Sample customer number, 1 to 2,357 (the `ID` used in the two summary files) |
| 3 | Purchase date, `YYYYMMDD` |
| 4 | Number of CDs |
| 5 | Dollar value |

## `data/raw/cdnow_customers_summary.csv`

One row per sample customer; 2,357 rows. Calibration period = the first 39 weeks. Used to check the repository's recency/frequency summary customer by customer.

| Field | Type | Meaning |
|---|---|---|
| `ID` | integer | Sample customer number (1 to 2,357) |
| `frequency` | integer | Repeat purchases (purchases after the first) in the calibration period; 0 to 29 |
| `recency` | number | Weeks between first and last purchase in the calibration period; 0 when there is no repeat purchase |
| `T` | number | Weeks between first purchase and the end of the calibration period; 27.00 to 38.86 |

## `data/raw/cdnow_customers_summary_with_transactions.csv`

As above, with one extra field; 2,357 rows. Used to check the Gamma-Gamma spend model.

| Field | Type | Meaning |
|---|---|---|
| `ID` | integer | Sample customer number |
| `x` | integer | Repeat purchases (same quantity as `frequency`) |
| `t_x` | number | Recency in weeks (same quantity as `recency`) |
| `T` | number | Customer age in weeks |
| `zbar` | number | Mean dollar value of the customer's repeat purchases; 0 when there are none; maximum 299.63 |

## Fixity

File sizes and SHA-256 checksums are recorded in `data/datapackage.json`. Run `python scripts/check_metadata.py` to confirm that the files on disk still match them and that every documented field exists in the data.
