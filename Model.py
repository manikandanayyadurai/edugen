import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1️⃣ Load Both Datasets
# =========================

mat = pd.read_csv("student-mat.csv", sep=';')
por = pd.read_csv("student-por.csv", sep=';')

# Add subject column
mat["subject"] = "math"
por["subject"] = "portuguese"

# Combine both
data = pd.concat([mat, por], ignore_index=True)

print("Total Records:", len(data))

# =========================
# 2️⃣ Clean Data
# =========================

# Convert yes/no to 1/0
binary_cols = [
    "schoolsup","famsup","paid","activities","nursery",
    "higher","internet","romantic"
]

for col in binary_cols:
    data[col] = data[col].map({"yes":1, "no":0})

# =========================
# 3️⃣ Encode Categorical Variables
# =========================

categorical_cols = [
    "school","sex","address","famsize","Pstatus",
    "Mjob","Fjob","reason","guardian","subject"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# =========================
# 4️⃣ Define Features & Target
# =========================

X = data.drop("G3", axis=1)
y = data["G3"]

# Drop G1 and G2 if you want harder prediction
# But keeping them improves accuracy
# (You can experiment later)

# =========================
# 5️⃣ Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6️⃣ Train Model
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 7️⃣ Evaluate Model
# =========================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Performance:")
print("MAE:", round(mae,2))
print("R2 Score:", round(r2,2))

# =========================
# 8️⃣ Save Model
# =========================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(label_encoders, open("encoders.pkl", "wb"))

print("Model saved as model.pkl")