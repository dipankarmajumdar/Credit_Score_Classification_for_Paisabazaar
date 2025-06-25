# Credit Score Classification for Paisabazaar

## 🔍 Problem Statement

Paisabazaar, a leading fintech platform, wants to predict the **Credit Score category** (`Poor`, `Standard`, `Good`) of its customers based on their financial profile. Accurate predictions can help the platform:

- Recommend appropriate credit products
- Reduce loan default risks
- Improve customer segmentation and targeting

This project aims to develop a **robust classification model** that categorizes users into credit score groups using structured data.

---

## 📂 Dataset

The dataset is sourced from **Google Drive** and loaded via `gdown`.

```python
import gdown
file_id = "1llz87ojrAO0kq4SUaoX1v-Kkxe5qIJ-R"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, 'dataset.csv', quiet=False)

# Load
import pandas as pd
df = pd.read_csv('dataset.csv')
```

---

## 🔧 Key Steps

### 1. Know Your Data

- Loaded & explored the dataset
- Checked for missing values, unique values, outliers
- Cleaned and made it ready for analysis

### 2. Data Visualization & Insights

Used **15 meaningful charts** using the **UBM Rule** (Univariate, Bivariate, Multivariate):

- Income, age, occupation, monthly balance distributions
- Heatmaps and correlation plots
- Found strong correlation between `Outstanding_Debt`, `Credit_Utilization_Ratio`, and Credit Score

### 3. Data Preprocessing

- Missing value check: **No imputation needed**
- Encoding: Used `OneHotEncoder` via `ColumnTransformer`
- Scaling: StandardScaler used
- Outliers: Treated using **IQR** method
- High-cardinality column `Type_of_Loan` dropped before SMOTE due to memory issues

### 4. Handling Imbalanced Data

- Used **SMOTE (Synthetic Minority Oversampling Technique)** after splitting to balance the classes
- Verified distribution using `collections.Counter`

---

## 🧠 Machine Learning Models

| Model                  | Accuracy   | Macro F1 | Final? |
| ---------------------- | ---------- | -------- | ------ |
| RandomForestClassifier | 0.8125     | 0.80     | ❌     |
| XGBoostClassifier      | **0.8224** | **0.82** | ✅     |
| LightGBM               | 0.7958     | 0.79     | ❌     |

✅ **Final Model: XGBoostClassifier (after RandomizedSearchCV tuning)**

---

## ✅ Evaluation Metrics

- **Accuracy**: 82.24%
- **Precision (Macro Avg)**: 81%
- **Recall (Macro Avg)**: 82%
- **F1-Score (Macro Avg)**: 82%
- **Confusion Matrix**:

  ```
  [[2862    4  700]
   [  15 4874  910]
   [ 734 1189 8712]]
  ```

🔍 Interpretation:

- Most "Good" and "Standard" credit customers were predicted correctly.
- Model slightly confuses “Standard” and “Good” classes, which is acceptable from a business standpoint.

---

## 🔬 Feature Importance (XGBoost)

Top Features:

- **Monthly_Balance**
- **Annual_Income**
- **Credit_Utilization_Ratio**
- **Outstanding_Debt**
- **Number_of_Loan**
- **Num_Bank_Accounts**

Used XGBoost's `feature_importances_` plot to explain model interpretability.

---

## 🏁 Conclusion

- The model provides a **reliable classification system** for credit scores based on financial behavior.
- After extensive preprocessing, visualization, and model comparison, **XGBoost** emerged as the most accurate and business-aligned model.
- This model can be deployed to help **Paisabazaar** optimize its credit offerings and customer targeting.

---

## 📦 Project Structure

```
.
├── Credit_Score_Classification_for_Paisabazaar.ipynb
├── dataset.csv  (via gdown)
├── README.md
```

---

## 👤 Author

**Dipankar Majumdar**
[GitHub](https://github.com/dipankarmajumdar) | [LinkedIn](https://www.linkedin.com/in/dipankarm9)

---

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).
