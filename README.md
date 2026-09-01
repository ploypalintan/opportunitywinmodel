# Opportunity Win Probability Model

A machine learning project exploring whether observable CRM attributes can support capacity-constrained prioritisation of sales opportunities, built to help a presales or sales team decide where to focus limited time.

## Why This Exists

Presales and sales teams can't give every open opportunity equal attention. This project asks a direct question: **can we use standard CRM data to rank opportunities by how likely they are to close, so effort goes where it matters most?**

## Data

Four relational CRM tables, joined to one row per opportunity:
- `sales_pipeline.csv` — opportunity outcomes, dates, price (fact table)
- `accounts.csv` — account attributes (industry, revenue, employee count)
- `products.csv` — product line attributes
- `sales_teams.csv` — sales rep and regional context

Target variable: deal outcome (Won = 1, Lost = 0). The dataset is moderately imbalanced (63.2% Won / 36.8% Lost).

## Approach

A staged, leakage-safe modelling workflow:

1. **Data integration** — joined all four tables to opportunity-grain, validated relational integrity and data quality.
2. **Leakage-safe feature selection** — explicitly excluded any field only known *after* a deal closes (`close_date`, `close_value`), since the model needs to be useful *before* an outcome is known.
3. **Baseline modelling** — logistic regression and random forest, evaluated on AUC.
4. **Feature engineering** — added relative deal-size signals (deal value vs. account revenue, vs. employee count), sector-price interaction terms, and revenue banding, to capture economic proportionality a raw price figure misses.
5. **Model interpretation** — used SHAP (SHapley Additive exPlanations) to understand which features actually drove predictions, and in which direction, not just which ones the model leaned on.
6. **Reframing** — when initial results showed limited raw discrimination, reframed the objective from "classify Won vs. Lost" to "rank opportunities by relative likelihood," a genuinely different and more honest use of a model like this.

## Results, Stated Plainly

Raw classification performance was modest: **AUC ≈ 0.51–0.53** across both logistic regression and random forest. That's close to chance-level separability using firmographic and pricing data alone.

What the model *did* show real value in was ranking: **Precision@10% and Lift@10% ≈ 1.11**, meaning the top 10% of opportunities the model flagged as highest-likelihood converted roughly 11% better than the baseline win rate. That's a real, usable signal for prioritisation, even though the model isn't a reliable yes/no classifier.

## Honest Limitations

The dataset only contains firmographic and pricing attributes, no behavioral or process-level signals: number of meetings, touchpoint frequency, time between pipeline stages, objections raised, or competitive context. Conversion in B2B sales is often driven by exactly this kind of execution data, which this dataset doesn't capture. The model's modest AUC reflects that real constraint, not a flaw in the modelling approach itself.

## Repository Contents

- `capstone_ploypalin_tan.ipynb` — full analysis notebook: data integration, EDA, modelling, feature engineering, SHAP interpretation
- `Capstone_Ploypalin_Tan_Slides.pdf` — summary presentation of findings

## Built With

Python · pandas · scikit-learn · SHAP · matplotlib

## Context

Built as a capstone project for General Assembly's Data Analytics Bootcamp (Nov 2025 – Mar 2026).
