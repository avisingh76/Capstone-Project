# Part 2 — Regression & Classification Models

## Required documents
- Cleaned dataset (`data/cleaned_data.csv`) carried over from Part 1.
- Code file (`modelling.ipynb`).
- README.md (this file).

## Task to do
- Define the feature matrix (X) and the two labels (regression and classification).
- Encode categorical columns appropriately (ordinal vs one-hot).
- Split data into train and test sets.
- Scale features without leaking test-set information into training.
- Train Linear Regression and Ridge Regression for the regression task.
- Train Logistic Regression for the classification task.
- Evaluate regression with MSE and R².
- Evaluate classification with a confusion matrix, precision, recall, F1, and ROC-AUC.
- Analyze how classification metrics change across decision thresholds.
- Compare a strongly regularized model (C=0.01) against the baseline (C=1.0).
- Compute a bootstrap confidence interval for the AUC difference between the two.

## Output result
- Regression metrics table (MSE, R²) for Linear Regression vs Ridge.
- Top 3 regression coefficients by magnitude, with interpretation.
- Classification confusion matrix and classification report.
- ROC-AUC score and interpretation.
- Threshold sensitivity table (0.30–0.70).
- Regularization comparison table (C=1.0 vs C=0.01).
- Bootstrap 95% confidence interval for the AUC difference.

## Dataset chosen
- Dataset name: California Housing.
- Source: scikit-learn California Housing dataset, with a derived categorical feature for EDA (carried over from Part 1).
- Why this dataset: it has more than 500 rows, several numeric columns, at least 2 categorical columns, and a clearly identifiable
  target variable (`median_house_value`) that supports both a regression task and, after a median split, a classification task
  (`expensive`).
- Rows: 20,640.
- Regression target: `median_house_value` (continuous).
- Classification target: `expensive` (binary, median split).

## Label definitions
- **Feature matrix (X):** all columns from `cleaned_data.csv` except `median_house_value` and `expensive` — 14 columns before
  encoding, 17 after one-hot expansion of `ocean_proximity`.
- **Regression label (y_reg):** `median_house_value`.
- **Classification label (y_clf):** `expensive`, already created in `preprocessing1.ipynb` as
  `(median_house_value > median_house_value.median()).astype(int)`.
  Class balance: `0` → 10,323 rows, `1` → 10,317 rows — near-perfectly balanced by construction.
- `median_house_value` is dropped entirely from X to avoid leakage: leaving it in would let the classifier trivially learn
  `expensive`, and it cannot be a regression feature since it is the regression target itself.

## Categorical encoding
- `income_category` (`Low < Medium < High < Very High`) has a natural order, so it is **label encoded**:
  `{Low: 0, Medium: 1, High: 2, Very High: 3}`.
- `ocean_proximity` has no natural order, so it is **one-hot encoded** (`pd.get_dummies(..., drop_first=True)`), avoiding a false
  ordinal relationship that label encoding would otherwise imply. `<1H OCEAN` is dropped as the reference category; remaining
  dummies are `ocean_proximity_INLAND`, `ocean_proximity_ISLAND`, `ocean_proximity_NEAR BAY`, `ocean_proximity_NEAR OCEAN`.

## Data leakage prevention
- `StandardScaler` is fit **only on `X_train`**, then used to transform both `X_train` and `X_test`.
- Fitting on the combined train+test data would leak test-set statistics into training, producing an overly optimistic and
  invalid performance estimate.

## Regression results — Linear Regression vs Ridge

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 4,177,100,818.18 | 0.6812 |
| Ridge (alpha=1.0) | 4,176,963,555.16 | 0.6812 |

- Top 3 features by absolute coefficient value: `latitude` (−53,461.27), `longitude` (−53,299.10), `population` (−40,287.79).
- Ridge and OLS are nearly identical here because `alpha=1.0` applies only mild shrinkage relative to the coefficient scale — the
  correlated features (`total_rooms`, `households`, `rooms_per_household`) aren't causing severe instability at this alpha.

## Classification results — Logistic Regression (C=1.0)
- Training class balance: 50.06% expensive vs 49.94% not expensive — no imbalance handling needed.
- Confusion matrix (test set, threshold 0.5): `[[1739, 338], [328, 1723]]`.
- Precision / Recall / F1 (both classes): 0.84.
- Overall accuracy: 0.84.
- ROC-AUC: 0.9155 — the model ranks a random "expensive" house above a random "not expensive" house 91.55% of the time.
- Given the near-identical precision/recall, **recall** is the priority if this model is used to flag houses for a premium-pricing
  review, since missing a genuinely expensive house is costlier than an unnecessary review.

## Threshold sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.7551 | 0.9186 | 0.8289 |
| 0.40 | 0.7987 | 0.8840 | 0.8392 |
| 0.50 | 0.8360 | 0.8401 | 0.8380 |
| 0.60 | 0.8744 | 0.7840 | 0.8267 |
| 0.70 | 0.8993 | 0.7094 | 0.7931 |

- F1-maximizing threshold: **0.40** (F1 = 0.8392), narrowly ahead of the default 0.50.
- Lowering the threshold favors recall; raising it favors precision.

## Regularization experiment — C=1.0 vs C=0.01

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.8360 | 0.8401 | 0.9155 |
| C=0.01 (strong regularization) | 0.8107 | 0.8415 | 0.9095 |

- Smaller `C` means a stronger L2 penalty. Dropping `C` to 0.01 slightly worsened precision and AUC, suggesting the baseline model
  wasn't overfitting much — the stronger penalty mostly suppressed useful signal.

## Bootstrap confidence interval for AUC difference
- 500 bootstrap resamples of the test set (seed=42).
- Mean AUC difference (C=1.0 minus C=0.01): **0.0060**.
- 95% CI: **[0.0037, 0.0085]** — excludes zero, so the C=1.0 model's small AUC advantage is statistically consistent across
  resamples, not a fluke of one split.

## Final files
- `modelling.ipynb`
- `README_Part2.md`