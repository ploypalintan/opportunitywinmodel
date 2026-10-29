# Opportunity Win Probability Model

A machine learning project exploring whether observable CRM attributes can support capacity-constrained prioritisation of sales opportunities, built to help a presales or sales team decide where to focus limited time.

## Why This Exists

Presales and sales teams can't give every open opportunity equal attention. This project asks a direct question: **can we use standard CRM data to rank opportunities by how likely they are to close, so effort goes where it matters most?**

**Short answer: not reliably, with this data.** The results below explain why, and what data would be needed.

## Data

Four relational CRM tables, joined to one row per opportunity:
- `sales_pipeline.csv` — opportunity outcomes, dates, price (fact table)
- `accounts.csv` — account attributes (industry, revenue, employee count)
- `products.csv` — product line attributes
- `sales_teams.csv` — sales rep and regional context

Target variable: deal outcome (Won = 1, Lost = 0). The dataset is moderately imbalanced (63.2% Won / 36.8% Lost). The models were trained on 5,033 opportunities and evaluated on a held-out test set.

## Approach

A staged, leakage-safe modelling workflow:

1. **Data integration** — joined all four tables to opportunity-grain, validated relational integrity and data quality.
2. **Leakage-safe feature selection** — explicitly excluded any field only known *after* a deal closes (`close_date`, `close_value`), since the model needs to be useful *before* an outcome is known.
3. **Baseline modelling** — logistic regression and random forest, evaluated on AUC and on ranking quality (Precision@10% and Lift@10%).
4. **Feature engineering** — added relative deal-size signals (deal value vs. account revenue, vs. employee count), sector-price interaction terms, and revenue banding, to capture economic proportionality a raw price figure misses.
5. **Model interpretation** — used SHAP (SHapley Additive exPlanations) to see which features drove predictions, and in which direction.
6. **Reframing** — when initial results showed no raw discrimination, the objective was reframed from "classify Won vs. Lost" to "rank opportunities by relative likelihood".

## Results, Stated Plainly

All figures below are computed in the notebook, on a single train/test split.

| Model | AUC | Precision@10% | Lift@10% |
|---|---|---|---|
| Logistic regression (baseline) | 0.487 | 0.551 | 0.87 |
| Random forest (baseline) | 0.486 | not computed | not computed |
| Logistic regression, engineered features (v2) | 0.502 | 0.672 | 1.08 |
| Random forest, engineered features (v3) | 0.505 | 0.582 | 0.94 |

- **Classification:** AUC is about 0.49–0.51 for every model. That is chance level. With these features, the models cannot tell won deals from lost ones.
- **Ranking:** Only one configuration, logistic regression with engineered features, beat the baseline win rate in the top 10% of ranked deals (lift 1.08, roughly 8% better than baseline). The random forest with the same features did not (lift 0.94, below baseline). The top 10% is only about 130 deals, so this one result could be noise.
- **What drove predictions (SHAP):**
