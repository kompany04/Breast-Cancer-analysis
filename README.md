#  Breast Cancer Detection using Feature Selection and Neural Network (MLP)

This project builds a **machine learning pipeline** for breast cancer classification using the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`. The pipeline includes **exploratory data analysis**, **feature selection**, and **neural network classification** using `MLPClassifier`. The final model achieves high accuracy and is saved as a `.pkl` file for future use.

---

## 🔍 Project Workflow

1. **Dataset Loading**
   - Uses `load_breast_cancer()` from `sklearn.datasets`.
   - Converts the data into a `pandas` DataFrame for analysis.

2. **Exploratory Data Analysis (EDA)**
   - Heatmap of feature correlations using `seaborn`.
   - Histograms of feature distributions.

3. **Feature Selection**
   - Implements `SelectKBest` with `f_classif` to select top 10 most informative features.
   - Selected features include:
     - `mean radius`
     - `mean perimeter`
     - `mean area`
     - `mean concavity`
     - `mean concave points`
     - `worst radius`
     - `worst perimeter`
     - `worst area`
     - `worst concavity`
     - `worst concave points`

4. **Evaluation**
   - Accuracy on test set: **97.3%**
   - Classification report shows high precision, recall, and F1-score.
   - Confusion matrix visualized with `seaborn`.

5. **Model Export**
   - Best model saved using `joblib` as `gridsearch.pkl`.

---

## 📁 Files

| File                  | Description                              |
|-----------------------|------------------------------------------|
| `breast_cancer.py/ipynb` | Main notebook/script with full pipeline |
| `gridsearch.pkl`      | Saved best model from `GridSearchCV`     |
| `README.md`           | Project overview and structure           |

---

## 🛠️ Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `joblib`

---

## ✅ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook/script
python breast_cancer.py
