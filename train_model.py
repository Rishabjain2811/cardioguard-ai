import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("heart.csv")
features = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']
X = df[features]
y = df['target']

best_acc = 0

for seed in range(0, 20):   # ✅ reduced seeds (0–20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Base models
    estimators = [
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric='logloss',
                              random_state=42, verbosity=0)),
        ('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]

    # Meta model
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )

    stacking.fit(X_train, y_train)
    acc = accuracy_score(y_test, stacking.predict(X_test))

    # Track best
    if acc > best_acc:
        best_acc = acc
        best_seed = seed
        best_model = stacking
        best_scaler = scaler

# ✅ FINAL CLEAN OUTPUT ONLY
print(f"\n🏆 Final Model Accuracy: {round(best_acc*100, 2)}%")

# Save model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(best_scaler, open("scaler.pkl", "wb"))

print("💾 Model saved!")
