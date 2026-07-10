# Part 1 — Data Cleaning, EDA and Preprocessing

## Setup
```
  pip install -r requirements.txt
```

## Required documents
- Raw dataset (`data/housing_raw.csv`), code files (`eda1.ipynb`, `preprocessing1.ipynb`), cleaned dataset
  (`data/cleaned_data.csv`), visualizations, README.md.

## Dataset chosen
- **California Housing** (1990 Census), 20,640 rows. Meets all criteria: >500 rows, multiple numeric columns, 2
  categorical columns (`ocean_proximity`, `income_category`), and a target (`median_house_value`) usable for both
  regression and classification (`expensive`, via median split).

---

## Part A — EDA (`eda1.ipynb`)

- **Missing values:** only `total_bedrooms` (207 rows, 1.0%) — flagged for imputation, not dropped.
- **Duplicates:** none found.
- **Skewness:** `population` is most skewed (≈4.94), along with `total_rooms`, `total_bedrooms`, `households` — all
  right-skewed, so **median** is preferred over mean for imputation.
- **Outliers (IQR):** found in `total_rooms` (plausible large block groups) and `median_house_value` (top-coding cap
  at $500,001) — both **retained**, not dropped.
- **Visualizations:** line plot (value over row order), bar chart (mean value by `ocean_proximity`), histogram
  (`population` distribution), scatter plot (`median_income` vs `median_house_value` — clear positive relationship,
  flattens near the cap), box plot (value by `ocean_proximity` — `INLAND` lowest, `NEAR BAY`/`NEAR OCEAN` higher).
- **Correlation:** `total_rooms` & `households` most correlated pair (likely due to block-group size, not causation).
  Spearman preferred over Pearson for Part 2 feature selection, since data is skewed.
- **Grouped aggregation:** `NEAR BAY` has the highest reliable mean and std by `ocean_proximity`; highest/lowest
  reliable group mean ratio is ~2×, showing real predictive signal.
- **Output:** saved `data/housing_clean.csv`.

## Part B — Preprocessing (`preprocessing1.ipynb`)

- **Imputation:** confirmed median > mean for skewed columns; filled `total_bedrooms` using the median **within each
  `ocean_proximity` group**.
- **Cleanup:** dropped any duplicates, fixed count columns to `int64`, converted `ocean_proximity` to `category`
  (reduced memory usage).
- **`is_value_capped`:** binary flag for the 965 rows (4.7%) at the $500,001 reporting ceiling — kept, not dropped.
- **Engineered ratios:** `rooms_per_household`, `bedrooms_per_room`, `population_per_household` — reduce block-group
  size effects and multicollinearity.
- **`income_category`:** ordinal bins from `median_income` (Low / Medium / High / Very High).
- **`expensive` target:** binary, median split on `median_house_value`. Class balance: 10,323 (0) vs 10,317 (1) — near
  perfectly balanced, no resampling needed.
- **Output:** saved final feature-engineered dataset to `data/cleaned_data.csv`, used by Parts 2–4.

## Final files
- `eda1.ipynb`,
- `preprocessing1.ipynb`,
- `data/housing_clean.csv`,
- `data/cleaned_data.csv`,
- `README_Part1.md`
