import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("student_data.csv")

df["study_hours"] = df["study_hours"].fillna(df["study_hours"].mean())
df["attendance"] = df["attendance"].fillna(df["attendance"].mean())

df = df.drop_duplicates()

encoder = LabelEncoder()
df["result_encoded"] = encoder.fit_transform(df["result"])

X = df[["study_hours", "attendance"]]
y = df["result_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)

report = classification_report(y_test, prediction)
print("Classification Report:")
print(report)

joblib.dump(model, "student_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Model saved successfully.")