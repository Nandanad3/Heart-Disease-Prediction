import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("data/heart.csv")

# ==========================
# HANDLE MISSING VALUES
# ==========================

num_cols = [
    'trestbps',
    'chol',
    'thalch',
    'oldpeak',
    'ca'
]

for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

cat_cols = [
    'sex',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal',
    'dataset'
]

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# ==========================
# TARGET CONVERSION
# ==========================

df['num'] = df['num'].apply(
    lambda x: 0 if x == 0 else 1
)

# ==========================
# DROP ID
# ==========================

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# ==========================
# ENCODING
# ==========================

categorical_columns = [
    'sex',
    'cp',
    'restecg',
    'slope',
    'thal',
    'dataset'
]

available_cols = [
    col for col in categorical_columns
    if col in df.columns
]

df = pd.get_dummies(
    df,
    columns=available_cols,
    drop_first=True
)

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop('num', axis=1)
y = df['num']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# LOAD SAVED MODEL
# ==========================

with open(
    "models/heart_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# ACCURACY
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ==========================
# CLASSIFICATION REPORT
# ==========================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)