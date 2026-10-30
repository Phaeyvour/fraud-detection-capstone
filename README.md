# Credit Card Fraud Detection
A machine learning project that detects fraudulent credit card transactions in a highly imbalanced dataset. Built as the capstone project for the Gomycode Data Science programme.

## Overview
Fraud makes up a tiny share of all transactions, so a model can look accurate while missing most fraud. This project explores the data, compares several tree-based models and evaluates them with ROC-AUC, which suits imbalanced data better than accuracy.

## Dataset
- **Source:** [Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns
- **Fraud cases:** 492 (about 0.17%)
- **Features:** `Time`, `Amount`, and 28 anonymised features (`V1`–`V28`) produced by PCA
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- **Data quality:** No missing values

## Approach
1. **Data checks:** Checked for missing values and looked at the class imbalance.
2. **Exploratory analysis:**
   - Fraud vs. legitimate transactions by hour of day
   - Amount distributions by class (total, mean, median, min, max)
   - Correlation heatmap and feature distributions by class
3. **Data split:** Train, validation and test sets (random state 2018).
4. **Modelling:** Trained and compared five models:
   - Random Forest
   - AdaBoost
   - CatBoost
   - XGBoost
   - LightGBM (with 5-fold cross-validation)
5. **Imbalance handling:** Used `scale_pos_weight` in LightGBM to give fraud cases more weight.
6. **Evaluation:** ROC-AUC, confusion matrices and feature importance plots.

## Results
| Model | ROC-AUC | Evaluated on |
|---|---|---|
| Random Forest | 0.853 | Validation set |
| AdaBoost | 0.833 | Validation set |
| CatBoost | 0.858 | Validation set |
| **XGBoost** | **0.979** | Test set |
| LightGBM (5-fold CV) | 0.976 | Out-of-fold, training data |

XGBoost gave the strongest results on the held-out test set.

## Tools and Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, XGBoost, LightGBM, CatBoost

## How to Run
1. Download `creditcard.csv` from the Kaggle link above.
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm catboost
   ```
3. Update the file path in the notebook to point to `creditcard.csv`.
4. Run the notebook from top to bottom.

## Team
Built by a 4-member team at Gomycode.

## Possible Improvements
- Report precision, recall and F1, which matter more than accuracy for fraud detection
- Try resampling methods such as SMOTE
- Tune the decision threshold to balance missed fraud against false alarms
