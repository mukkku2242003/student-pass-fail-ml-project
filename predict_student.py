import pandas as pd
import joblib

model = joblib.load("student_model.pkl")
encoder = joblib.load("label_encoder.pkl")

new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [82]
})

prediction = model.predict(new_student)

final_result = encoder.inverse_transform(prediction)

print("Student Result:", final_result[0])