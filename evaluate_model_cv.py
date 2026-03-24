import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("heart.csv")

# Features and target
features = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak',
    'slope','ca','thal'
]

X = data[features]
y = data['target']

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", round(scores.mean()*100,2), "%")
print("Standard Deviation:", round(scores.std()*100,2), "%")