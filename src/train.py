# src/train.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==========================
# 1. LOAD DATASET
# ==========================

df = pd.read_csv("data/heart.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)


# ==========================
# 2. HANDLE MISSING VALUES
# ==========================

# Numerical columns
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

# Categorical columns
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

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ==========================
# 3. CONVERT TARGET VARIABLE
# ==========================

# 0 = No Disease
# 1,2,3,4 = Disease

df['num'] = df['num'].apply(
    lambda x: 0 if x == 0 else 1
)

print("\nTarget Distribution:")
print(df['num'].value_counts())


# ==========================
# 4. DROP UNNECESSARY COLUMN
# ==========================

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)


# ==========================
# 5. ONE HOT ENCODING
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
# 6. FEATURES AND TARGET
# ==========================

X = df.drop('num', axis=1)

y = df['num']

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================
# 7. TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================
# 8. TRAIN MODEL
# ==========================

model = LogisticRegression(
    max_iter=2000
)

model.fit(X_train, y_train)

print("\nModel Training Completed")


# ==========================
# 9. PREDICTIONS
# ==========================

y_pred = model.predict(X_test)


# ==========================
# 10. ACCURACY
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")


# ==========================
# 11. SAVE MODEL
# ==========================

with open(
    "models/heart_model.pkl",
    "wb"
) as file:

    pickle.dump(model, file)

print("\nModel Saved Successfully")
print("Location: models/heart_model.pkl")