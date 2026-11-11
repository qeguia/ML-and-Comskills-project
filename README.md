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

**Main findings:** Discount is the main risk factor, but its effect varies by category. Furniture losses are driven by discount depth (Tables and Bookcases are worst). Office Supplies losses are driven by which product is sold (Storage, Supplies). Technology is relatively resilient, Copiers and Machines stay profitable even at moderate discounts. A single discount policy is suboptimal; category-specific thresholds reduce losses

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/qeguia/ML-and-Comskills-project.git
cd ML-and-Comskills-project
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

```bash
python 01_eda.py                 # Explore the data
python 02_global_model.py        # Logistic regression baseline
python 03_category_models.py     # Separate LR per category
python 04_model_comparison.py    # LR vs Random Forest vs Gradient Boosting
python 05_feature_importance.py  # SHAP analysis
python 06_temporal_split.py      # Train on 2014-16, test on 2017
python 07_rf_per_category.py     # 3 Random Forests (professor's feedback)
python 08_discount_thresholds.py # Category-specific discount limits
python 09_error_analysis.py      # False positive/negative analysis
python 10_final_evaluation.py    # Final test set evaluation (run ONCE)
```

What each script does:

| Script | Purpose | Key Output |
|--------|---------|------------|
| `01_eda.py` | Summary stats, discount × category bar chart | `figures/eda_discount_by_category.png` |
| `02_global_model.py` | Logistic regression on all data | F1 ≈ 0.705, AUC ≈ 0.978 |
| `03_category_models.py` | 3 separate logistic regressions | Office Supplies F1 improves 0.316 → 0.400 |
| `04_model_comparison.py` | LR vs RF vs GBT comparison | GBT best: F1 ≈ 0.745 |
| `05_feature_importance.py` | SHAP per-transaction explanations | `figures/shap_per_category.png` |
| `06_temporal_split.py` | Robustness: train past, test future | F1 drops only 0.016 |
| `07_rf_per_category.py` | 3 Random Forests with ordinal encoding | `figures/rf_feature_importance_per_category.png` |
| `08_discount_thresholds.py` | Find max safe discount per category | `figures/discount_thresholds.png` |
| `09_error_analysis.py` | Analyze where the model fails | False positive/negative breakdown |
| `10_final_evaluation.py` | Final evaluation on held-out test set | F1 with 95% confidence intervals |

---

## Project Structure

```
ML-and-Comskills-project/
├── data.csv
├── environment.yml
├── README.md
├── figures/
│   ├── eda_discount_by_category.png
│   ├── shap_per_category.png
│   ├── rf_feature_importance_per_category.png
│   └── discount_thresholds.png
└── src/
    ├── data_loader.py
    ├── 01_eda.py
    ├── 02_global_model.py
    ├── 03_category_models.py
    ├── 04_model_comparison.py
    ├── 05_feature_importance.py
    ├── 06_temporal_split.py
    ├── 07_rf_per_category.py
    ├── 08_discount_thresholds.py
    ├── 09_error_analysis.py
    └── 10_final_evaluation.py
```


---

## Methodology

### Data Preparation
- Transactions with discount > 50% removed (100% unprofitable, so no learning signal)
- `Sales` excluded from features (data leakage, as it is computed simultaneously with Profit)
- Binary target: `Unprofitable = 1 if Profit < 0`
- Stratified split: 60% train / 20% validation / 20% test

### Encoding
- **One-hot encoding** for logistic regression (scripts 02–06)
- **Ordinal encoding** for Random Forest (scripts 07–10), trees handle categorical splits natively, resulting in less sparse data

### Models
| Model | Type | Why |
|-------|------|-----|
| Logistic Regression (L2) | Linear baseline | It has interpretable coefficients, and is fast |
| Random Forest | Nonlinear ensemble | Handles interactions, built-in feature importance; 100 trees, tuned via 5-fold CV |
| Gradient Boosting | Nonlinear ensemble | Best overall performance; 100 estimators, learning rate = 0.1, tuned via CV |
| SHAP | Model explanation method | TreeExplainer for Random Forest, outputs feature importance per category |

### Evaluation Metrics
- **F1 (minority class)** — balances precision and recall for unprofitable transactions
- **AUC-ROC** — measures separability across all thresholds
- Accuracy is reported but not used for model selection (misleading with 81% majority class)

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
