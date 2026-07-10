# Part 3 — Ensembles, Hyperparameter Tuning & Full ML Pipeline

## Setup
```
  pip install -r requirements.txt
```

## Required documents
- Cleaned dataset (`data/cleaned_data.csv`) carried over from Part 1.
- Train/test split and scaling logic carried over from Part 2.
- Code file (`ensemble_tuned_pipeline.ipynb`).
- Serialized best model (`best_model.pkl`).
- README.md (this file).

## Task to do
- Train an unconstrained Decision Tree and observe overfitting.
- Train a controlled Decision Tree (max_depth, min_samples_split) and compare the train/test gap.
- Compare Gini vs Entropy split criteria.
- Train a Random Forest and extract feature importances.
- Train a Gradient Boosting model and compare against the Random Forest.
- Run a feature ablation study on the lowest-importance features.
- Cross-validate all models with 5-fold StratifiedKFold on ROC-AUC.
- Tune the Random Forest with GridSearchCV.
- Produce a manual learning curve across training-set fractions.
- Serialize the best pipeline with joblib and confirm it reloads and predicts correctly.

## Output result
- Train/test accuracy for unconstrained vs controlled Decision Trees.
- Gini and Entropy formulas plus their comparison table.
- Random Forest feature importances (top 5) and test ROC-AUC.
- Gradient Boosting metrics compared against Random Forest.
- Feature ablation AUC comparison (full vs reduced feature set).
- 5-fold cross-validation mean/std AUC for all four models.
- GridSearchCV best parameters and best CV AUC.
- Learning curve table (5 training fractions) with a data-limited vs capacity-limited conclusion.
- `best_model.pkl` saved and reload-and-predict confirmation.
- Final summary comparison table with a model recommendation.

## Dataset chosen
- Dataset name: California Housing.
- Source: scikit-learn California Housing dataset, with a derived categorical feature for EDA (carried over from Part 1).
- Why this dataset: it has more than 500 rows, several numeric columns, at least 2 categorical columns, and a clearly identifiable
  target variable (`median_house_value`) that supports both a regression task and, after a median split, a classification task
  (`expensive`) — the same target used throughout Parts 2 and 3.
- Rows: 20,640.
- Target variable used in this Part: `expensive` (binary classification).

## Decision Tree baseline (unconstrained)

| Metric | Value |
|---|---|
| Train accuracy | 1.0000 |
| Test accuracy | 0.8440 |

- Perfect train accuracy with a meaningfully lower test score is a clear overfitting signal — the tree memorizes training noise
  because it splits greedily with no depth limit.

## Controlled Decision Tree (max_depth=5, min_samples_split=20)

| Tree | Train Acc | Test Acc | Gap |
|---|---|---|---|
| Unconstrained | 1.0000 | 0.8440 | 0.1560 |
| Controlled (depth=5) | 0.8394 | 0.8268 | 0.0126 |

- `max_depth` limits how many levels of splits are allowed; `min_samples_split=20` stops splits on tiny, noisy subsets.
- The controlled tree's train/test gap (1.3 points) is far smaller than the unconstrained tree's (15.6 points) — clear evidence of
  reduced overfitting, even though raw test accuracy is slightly lower.

## Gini vs Entropy (both at max_depth=5)
- Gini impurity: `1 − Σ pᵢ²`
- Entropy: `−Σ pᵢ · log₂(pᵢ)`

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.8268 |
| Entropy | 0.8367 |

- Entropy edged out Gini slightly; in practice the two usually produce very similar trees.

## Random Forest (n_estimators=100, max_depth=10)

| Metric | Value |
|---|---|
| Train accuracy | 0.9207 |
| Test accuracy | 0.8840 |
| Test ROC-AUC | 0.9524 |

- Top 5 features by importance: `ocean_proximity_INLAND` (0.1991), `median_income` (0.1806), `income_category` (0.1617),
  `population_per_household` (0.1015), `longitude` (0.0870).
- Random Forest importance is the average Gini-impurity reduction a feature produces across all trees and splits — it has no
  sign and doesn't assume a functional form, unlike a regression coefficient.
- Bagging: each tree trains on a bootstrap sample and considers only a random subset of features per split, so individual trees
  make different, largely uncorrelated mistakes that average out.

## Gradient Boosting (n_estimators=100, learning_rate=0.1, max_depth=3)

| Metric | Value |
|---|---|
| Train accuracy | 0.8922 |
| Test accuracy | 0.8852 |
| Test ROC-AUC | 0.9533 |

- Slightly edges out the Random Forest on both accuracy and AUC, with a much smaller train/test gap (0.0070 vs 0.0367).

## Feature ablation study
- 5 lowest-importance features removed: `households`, `is_value_capped`, `ocean_proximity_NEAR OCEAN`,
  `ocean_proximity_NEAR BAY`, `ocean_proximity_ISLAND`.

| Model | Test AUC |
|---|---|
| Full model | 0.9524 |
| Reduced model | 0.9517 |

- AUC barely changes (−0.0007), so these 5 features are close to uninformative for this task; a simpler model could be deployed
  with negligible accuracy cost.

## Cross-validated comparison (5-fold StratifiedKFold, ROC-AUC)

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.9257 | 0.0020 |
| Decision Tree (max_depth=5) | 0.9100 | 0.0021 |
| Random Forest | 0.9527 | 0.0019 |
| Gradient Boosting | 0.9554 | 0.0020 |

- Cross-validation gives both a stable mean estimate and a standard deviation, which a single train/test split cannot provide.

## Hyperparameter tuning — GridSearchCV (Random Forest)
- Best parameters: `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200`.
- Best CV AUC: **0.9597**.
- Grid size: 3 × 3 × 2 = 18 configurations × 5 folds = 90 total model fits.
- Grid Search is exhaustive but scales multiplicatively; Randomized Search samples a fixed number of combinations, scaling
  better but without guaranteeing the single best combination is found.

## Manual learning curve (best tuned pipeline)

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 1.0000 | 0.9450 |
| 0.4 | 1.0000 | 0.9508 |
| 0.6 | 1.0000 | 0.9540 |
| 0.8 | 1.0000 | 0.9566 |
| 1.0 | 1.0000 | 0.9593 |

- Training AUC stays at 1.0000 throughout since `max_depth=None` lets the forest fit training data perfectly regardless of size.
- Test AUC keeps rising through 100% of available data with no sign of flattening.
- Conclusion: the model is currently **data-limited rather than capacity-limited** — more training data would likely help
  further.

## Model serialization
```python
joblib.dump(best_pipeline, "best_model.pkl", compress=9) #(to keep best_model.pkl file size under limit)
loaded_model = joblib.load("best_model.pkl")
sample_predictions = loaded_model.predict(X_test.iloc[:2])
# Output: [0 0]
```
- Reload-and-predict ran without errors, confirming the pipeline is fully serializable and portable.

## Summary comparison table — all models (Parts 2 & 3)

| Model | CV Mean AUC | CV Std AUC | Test AUC (single split) |
|---|---|---|---|
| Logistic Regression (C=1.0) | 0.9257 | 0.0020 | 0.9155 |
| Logistic Regression (C=0.01) | — | — | 0.9095 |
| Decision Tree (max_depth=5) | 0.9100 | 0.0021 | — |
| Random Forest (max_depth=10) | 0.9527 | 0.0019 | 0.9524 |
| Gradient Boosting | 0.9554 | 0.0020 | 0.9533 |
| **Tuned Random Forest (GridSearchCV)** | **0.9597** | — | **0.9593** |

**Recommendation:** the tuned Random Forest is recommended — it has the highest CV and test AUC of any model evaluated, and is
packaged as a single serialized `Pipeline` (imputing + scaling + model) that can be called with one `.predict()` line, with no
manual preprocessing required downstream.

## Final files
- `ensemble_tuned_pipeline.ipynb`
- `best_model.pkl`
- `README_Part3.md`
