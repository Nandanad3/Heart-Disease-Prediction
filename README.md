# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

Heart Disease Prediction is a Machine Learning project that predicts whether a person is at risk of heart disease based on various medical attributes such as age, blood pressure, cholesterol level, chest pain type, and other health indicators.

The project uses data preprocessing, exploratory data analysis (EDA), machine learning model training, evaluation, and deployment through a Streamlit web application.

---

## 🎯 Objective

The main objective of this project is to:

* Analyze heart disease data.
* Identify important health factors associated with heart disease.
* Build a machine learning model for prediction.
* Evaluate model performance using classification metrics.
* Deploy the model using a user-friendly web interface.

---

## 🗂️ Project Structure

```text
Heart-Disease-Prediction/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── heart_model.pkl
│   └── model_columns.pkl
│
├── app/
│   ├── app.py
│   └── style.css
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

---

## 📊 Dataset Description

The dataset contains patient medical information used for heart disease prediction.

### Features

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Age of patient                    |
| sex      | Gender                            |
| cp       | Chest pain type                   |
| trestbps | Resting blood pressure            |
| chol     | Cholesterol level                 |
| fbs      | Fasting blood sugar               |
| restecg  | Resting ECG results               |
| thalch   | Maximum heart rate achieved       |
| exang    | Exercise-induced angina           |
| oldpeak  | ST depression                     |
| slope    | Slope of peak exercise ST segment |
| ca       | Number of major vessels           |
| thal     | Thalassemia                       |
| num      | Target variable                   |

### Target Variable

* 0 → No Heart Disease
* 1 → Heart Disease

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset overview
* Missing value analysis
* Statistical summary
* Target variable distribution
* Age distribution
* Gender distribution
* Cholesterol analysis
* Blood pressure analysis
* Correlation analysis
* Outlier detection

### Libraries Used

* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### Missing Value Handling

* Numerical columns filled using median values.
* Categorical columns filled using mode values.

### Target Conversion

The target variable was converted into binary classes:

```python
0 = No Heart Disease
1 = Heart Disease
```

### Encoding

Categorical variables were converted into numerical format using One-Hot Encoding.

### Train-Test Split

```python
80% Training Data
20% Testing Data
```

---

## 🤖 Machine Learning Model

### Algorithm Used

* Logistic Regression

### Why Logistic Regression?

* Suitable for binary classification problems.
* Easy to understand and interpret.
* Efficient for structured datasets.

---

## 📈 Model Evaluation

The model was evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score

### Example Output

```text
Accuracy: 84%

Confusion Matrix:
[[TN FP]
 [FN TP]]
```

---

## 🖥️ Streamlit Web Application

A user-friendly Streamlit application was developed to allow users to:

* Enter patient details.
* Predict heart disease risk.
* View prediction results instantly.

### Features

* Interactive interface
* Responsive design
* Real-time prediction
* Easy to use

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle

### Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Nandanad3/Heart-Disease-Prediction.git
```

### Move Into Project Folder

```bash
cd Heart-Disease-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

### Train Model

```bash
python src/train.py
```

### Evaluate Model

```bash
python src/evaluate.py
```

### Predict Using Script

```bash
python src/predict.py
```

### Launch Web Application

```bash
streamlit run app/app.py
```

---

## 📷 Screenshots

Add screenshots here:

### Home Page

Insert screenshot

### Prediction Result

Insert screenshot

### Evaluation Output

Insert screenshot

---

## 🔮 Future Enhancements

* Random Forest Classifier
* XGBoost Classifier
* Risk Percentage Prediction
* User Authentication
* Database Integration
* Doctor Dashboard
* PDF Report Generation
* Cloud Deployment

---

## 📚 Learning Outcomes

Through this project, the following concepts were learned:

* Data Cleaning
* Data Visualization
* Feature Engineering
* Machine Learning Workflow
* Model Evaluation
* Streamlit Deployment
* Git and GitHub Version Control

---

## 👩‍💻 Author

**Nandana Dinesh A**

Machine Learning | Python | Data Science Enthusiast

---

## ⭐ Acknowledgement

This project was developed for learning and academic purposes to understand the complete Machine Learning project lifecycle, from data preprocessing to model deployment.
