# 🫀 Heart Disease Prediction using Naive Bayes & KNN

This project performs Exploratory Data Analysis (EDA), preprocessing, and prediction of heart disease using **Naive Bayes** and **K-Nearest Neighbors (KNN)** classifiers. It also demonstrates how to export the trained model using **pickle**.

---

## 📂 Dataset

- **File:** `heart.csv`
- **Target Variable:** `HeartDisease` (0 = No, 1 = Yes)
- **Features include:**
  - Age
  - Sex
  - ChestPainType
  - RestingBP
  - Cholesterol
  - FastingBS
  - RestingECG
  - MaxHR
  - ExerciseAngina
  - Oldpeak
  - ST_Slope

---

## 🔧 Project Structure

```
📁 Heart Disease Prediction/
├── main_NB.ipynb        # Jupyter Notebook with full EDA & ML pipeline
├── model.pkl            # Trained Naive Bayes model (saved using pickle)
├── heart.csv            # Dataset file
└── README.md            # Project description and setup
```

---

## 📊 Features

- Pie chart of heart disease distribution
- Chest pain analysis by heart disease status
- Age statistics and distribution analysis
- Label encoding of categorical variables
- Training and evaluation of:
  - Naive Bayes Classifier
  - K-Nearest Neighbors Classifier
- Export model for deployment

---

## 🚀 How to Run

1. **Clone the repository or download the files.**
2. Make sure you have the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook main_NB.ipynb
   ```
4. The model will be saved as `model.pkl`.

---

## 🧪 Sample Prediction

```python
sample = [[50,1,0,145,0,1,1,139,1,0.7,1]]
prediction = model.predict(sample)
```

---

## 📌 Notes

- Data is label encoded for model compatibility.
- Naive Bayes tends to work well with categorical features.
- You can change test samples or replace Naive Bayes with other classifiers.

---

## 📬 Contact

Feel free to reach out for feedback, improvements, or deployment help.
