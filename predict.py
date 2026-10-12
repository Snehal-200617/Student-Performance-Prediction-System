from pathlib import Path

import joblib
import pandas as pd

# Find the saved model file in the same folder as this script.
model_path = Path(__file__).resolve().parent / "student_grade_model.joblib"

# Load the already-trained model.
model = joblib.load(model_path)

# Enter one hypothetical student's inputs.
new_student = pd.DataFrame([{
    "studytime": 2,
    "absences": 4,
    "G1": 10,
    "G2": 11,
}])

# Predict the final grade and display it.
predicted_grade = model.predict(new_student)[0]
print(f"Predicted final grade: {predicted_grade:.1f}")