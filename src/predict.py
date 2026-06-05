import pickle
import pandas as pd

# Load model
with open("models/heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Create empty dataframe with same columns used during training
sample = pd.DataFrame(columns=model_columns)
sample.loc[0] = 0

# Example patient
sample["age"] = 63
sample["trestbps"] = 145
sample["chol"] = 233
sample["fbs"] = True
sample["thalch"] = 150
sample["exang"] = False
sample["oldpeak"] = 2.3
sample["ca"] = 0

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")