# Predicting Unprofitable Retail Transactions: Discount Risk Identification & Profitability (DRIP)

---

## Overview
This project investigates which retail transactions are likely to be unprofitable based on discount depth, product category, and operational factors. Using machine learning classification methods, we analyze 9,994 transactions from the Superstore dataset (Kaggle) to build predictive models that could inform discount policy decisions.

**Research Question:**  To what extent can product category features (furniture, office supplies, technology) predict whether a retail transaction will be unprofitable, and which factors (discount depth, product sub-category, geographic region) contribute most to profit loss?

The project integrates:
- Binary classification (profitable vs. unprofitable)
- Model comparison: Logistic Regression, Random Forest, Gradient Boosting
- Per-category modeling to capture category-specific loss drivers
- Feature importance analysis using SHAP values
- Temporal validation (train on 2014–2016, test on 2017)

**Main findings:** Different product categories lose money for different reasons. Furniture is driven by discount depth, Office Supplies by specific sub-categories (Storage, Supplies), and Technology shows profitability in certain sub-categories (Copiers, Machines). A one-size-fits-all discount policy is suboptimal.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/qeguia/ML-and-Comskills-project.git
cd drip-superstore
```

### 2. Create the environment using the .yml file

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate ml-project
```

---

## Usage

This project uses a `src/`-based structure. All scripts must be run from inside the `src/` directory to ensure imports work correctly:

```bash
cd src
```

From there, run the analysis scripts in sequence:

### 1. Exploratory Data Analysis
Generate descriptive statistics and the discount × category figure:

```bash
python 01_eda.py
```

**Output:**  
- Console: unprofitable rate by category, discount-level statistics  
- Figure: `figures/discount_by_category.png`

### 2. Global Logistic Regression Model
Train a single model on all data:

```bash
python 02_global_model.py
```

**Output:**  
- Validation F1 (unprofitable class)
- AUC-ROC
- Top 5 predictive features
- Per-category breakdown (reveals Office Supplies weakness)

### 3. Per-Category Models
Train separate logistic regression models for Furniture, Office Supplies, and Technology:

```bash
python 03_category_models.py
```

**Output:**  
- Three separate models with category-specific coefficients
- Shows that Furniture is discount-driven, Office Supplies is sub-category-driven

### 4. Model Comparison (LR vs RF vs GBT)
Compare linear and tree-based methods:

```bash
python 04_model_comparison.py
```

**Output:**  
- Side-by-side F1 and AUC comparison
- Determines if nonlinear models improve performance

### 5. Feature Importance (SHAP)
Generate interpretability analysis:

```bash
python 05_feature_importance.py
```

**Output:**  
- Figure: `figures/shap_summary.png` (3-panel plot showing per-category SHAP values)
- Shows *why* each model flags transactions as unprofitable

### 6. Temporal Split Robustness Check
Test if models trained on 2014–2016 generalize to 2017:

```bash
python 06_temporal_split.py
```

**Output:**  
- Temporal validation F1 and AUC
- Confirms model robustness over time


---
## Output Example

### Discount × Category Figure (EDA)

Transactions with discounts >30% have dramatically different unprofitable rates by category:
- **Furniture:** 80%+ unprofitable
- **Office Supplies & Technology:** ~60% unprofitable

This motivates the per-category modeling approach.

### Per-Category Model Coefficients

| Category | Top Loss Driver | Coefficient |
|----------|----------------|-------------|
| **Furniture** | Discount | β = 3.43 |
| **Office Supplies** | Sub-Category: Storage | β = 3.52 |
| **Technology** | Discount | β = 3.11 |

Insight: Office Supplies losses are driven more by *which items* are sold than by discount depth.

### Model Comparison Results

| Model | F1 (global) | AUC-ROC |
|-------|------------|---------|
| Logistic Regression | 0.705 | 0.978 |
| Random Forest | 0.730 | 0.985 |
| Gradient Boosting | 0.745 | 0.988 |

Tree-based models offer marginal improvement, but logistic regression is competitive and more interpretable.


---

## Project Structure

```
drip-superstore/
│
├── data.csv
│
├── src/
│   ├── data_loader.py
│   ├── 01_eda.py
│   ├── 02_global_model.py
│   ├── 03_category_models.py
│   ├── 04_model_comparison.py
│   ├── 05_feature_importance.py
│   └── 06_temporal_split.py
│
├── figures/
│   ├── discount_by_category.png
│   └── shap_summary.png
│
├── environment.yml
├── README.md
└── .gitignore
```


---

## Methodology

### Data Preprocessing
1. **Filter invalid discounts:** Remove transactions with `Discount > 0.5` (100% unprofitable, no learning signal)
2. **Feature engineering:**
   - `Shipping_Days = Ship_Date - Order_Date`
   - One-hot encode: `Category`, `Sub-Category`, `Segment`, `Ship_Mode`, `Region`
3. **Target definition:** `Unprofitable = 1[Profit < 0]`
4. **Train/Val/Test split:** 60% / 20% / 20%, stratified by target

### Models
- **Logistic Regression:** L2 regularization (`C=1.0`), scaled features
- **Random Forest:** 100 trees, tuned via 5-fold CV
- **Gradient Boosting:** 100 estimators, learning rate = 0.1, tuned via CV
- **SHAP:** TreeExplainer for Random Forest, outputs feature importance per category

### Evaluation Metrics
- **F1 (unprofitable class):** Primary metric (handles class imbalance)
- **AUC-ROC:** Threshold-independent performance
- **Accuracy:** Reported but not prioritized (81.3% majority baseline)

---

## Key Decisions & Issues Encountered

### Issue 1: Global model fails on Office Supplies
The single logistic regression achieves F1 = 0.705 overall, but only F1 = 0.316 for Office Supplies transactions. This category has only 4% unprofitable rate after filtering, so the model essentially gives up on it.

**Solution:** Train separate models per category (professor's feedback). The Office Supplies-only model improves to F1 = 0.400 and reveals that sub-categories (Storage, Supplies) are the key drivers, not discount.

### Issue 2: Discount band >50% is trivially unprofitable
100% of transactions with discounts >50% are unprofitable across all categories. Including this in the training set inflates model performance artificially.

**Solution:** Remove these transactions from the dataset. Models now learn the *ambiguous* boundary where discounts start to hurt profitability.

### Issue 3: Temporal vs random split
The proposal used random splitting, but retail data may have temporal trends (seasonal effects, pricing policy changes).

**Solution:** Implement temporal validation: train on 2014–2016, test on 2017. Results confirm model robustness (F1 degrades by <2%).

---

## Technologies

- python 3.11
- pandas
- scikit-learn
- matplotlib
- shap
- numpy

---

## Reproducibility

All analysis is version-controlled and documented in commit history. Key reproducibility practices:
- **Random seeds:** Fixed at `random_state=42` for all splits and models
- **Environment:** `environment.yml` specifies exact package versions
- **Modular code:** Shared preprocessing in `data_loader.py`, separate scripts per analysis step
- **Public repository:** Code, figures, and commit history visible at [GitHub](https://github.com/qeguia/ML-and-Comskills-project)

---

## Contributing

This is an academic project for: **Machine Learning** & **Communication Skills**

Contributions follow standard practices:
- Clear commit messages describing changes
- Ensure all scripts run without errors before committing
- Document any new dependencies in `environment.yml`

---

## License

This project is for academic use within IE University's Bachelor in Applied Mathematics program. The Superstore dataset is publicly available on Kaggle under its standard terms of use. No formal license has been defined for the analysis code.

## References

1. **Dataset:**  
   Vivek (2023). *Superstore Dataset.* Kaggle. [https://www.kaggle.com/datasets/vivek468/superstore-dataset-final](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

2. **Methodology:**  
   Grewal, D., Ailawadi, K. L., Gauri, D., Hall, K., Kopalle, P., & Robertson, J. R. (2011). Innovations in retail pricing and promotions. *Journal of Retailing*, 87, S43–S52.
