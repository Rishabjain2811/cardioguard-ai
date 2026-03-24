import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# ---------------- LOAD DATA ----------------
data = pd.read_csv("heart.csv")

print("Dataset shape:", data.shape)

# ---------------- FEATURES ----------------
features = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak',
    'slope','ca','thal'
]

X = data[features]
y = data['target']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=4   # ← CHANGE THIS NUMBER if needed
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)

# ---------------- TRAIN ----------------
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Final Model Accuracy:", round(accuracy * 100, 2), "%")

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully.")