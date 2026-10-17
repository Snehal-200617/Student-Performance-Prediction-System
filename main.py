from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error
import joblib

csv_path = Path("data") / "student-mat.csv"
students = pd.read_csv(csv_path, sep=";")
# path to the CSV inside project's data folder ,Read the CSV sep using ;

print(students.head())

# Show the number of rows and columns.
print("Dataset shape:", students.shape)

# Show the column names.
print("Column names:", students.columns.tolist())

# Show each column's data type and how many values it contains.
print("\n--- Column information ---")
students.info()

# Count blank cells in each column.
print("\n--- Missing values per column ---")
print(students.isna().sum())

# Count rows that are exact copies of another row.
print("\n--- Duplicate rows ---")
print(students.duplicated().sum())

# Show summary statistics for the columns.
print("\n--- Summary statistics ---")
print(students.describe(include="all"))


# Draw a histogram to see how final grades are distributed.
plt.hist(students["G3"], bins=11, edgecolor="black")

# Add labels 
plt.title("Distribution of Final Grades")
plt.xlabel("Final grade (G3)")
plt.ylabel("Number of students")

# Display the chart window.
plt.show()

# Create a 2-by-2 layout so we can compare four inputs with final grade.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Compare study-time category with final grades.
students.boxplot(column="G3", by="studytime", ax=axes[0, 0])
axes[0, 0].set_title("Final Grade by Study Time")
axes[0, 0].set_xlabel("Study time category")
axes[0, 0].set_ylabel("Final grade (G3)")

# Compare absences with final grades.
axes[0, 1].scatter(students["absences"], students["G3"], alpha=0.6)
axes[0, 1].set_title("Absences vs Final Grade")
axes[0, 1].set_xlabel("Number of absences")
axes[0, 1].set_ylabel("Final grade (G3)")

# Compare the first-period grade (G1) with the final grade.
axes[1, 0].scatter(students["G1"], students["G3"], alpha=0.6)
axes[1, 0].set_title("G1 vs Final Grade")
axes[1, 0].set_xlabel("G1")
axes[1, 0].set_ylabel("G3")

# Compare the second-period grade (G2) with the final grade.
axes[1, 1].scatter(students["G2"], students["G3"], alpha=0.6)
axes[1, 1].set_title("G2 vs Final Grade")
axes[1, 1].set_xlabel("G2")
axes[1, 1].set_ylabel("G3")

#  display all four charts with proper spacing
plt.tight_layout()
plt.show()

# Inputs the model will use to predict the final grade.
feature_columns = ["studytime", "absences", "G1", "G2"]

target_column = "G3"

X = students[feature_columns]

y = students[target_column]

print("\nInput columns:", X.columns.tolist())
print("Target column:", target_column)
print("Number of student records:", len(X))


# Split the inputs and answers into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Test records:", len(X_test))

# Create two models to compare.
models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
}

# Train each model and evaluate its predictions on the test records.
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # RMSE measures the typical prediction error in grade points.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # R-squared compares the model with a mean-based prediction.
    r2 = r2_score(y_test, predictions)

    print(f"{name}: RMSE = {rmse:.2f}, R² = {r2:.2f}")
    
    # Add "school" to the numeric inputs to try one categorical feature.
mixed_features = ["studytime", "absences", "G1", "G2", "school"]
X_mixed = students[mixed_features]
y_mixed = students["G3"]

# Split this version's data the same way as before.
X_train_mixed, X_test_mixed, y_train_mixed, y_test_mixed = train_test_split(
    X_mixed,
    y_mixed,
    test_size=0.20,
    random_state=42
)

# Numeric missing values are filled with the median.
numeric_preparation = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
])

# Text categories are filled with the most common category, then encoded.
category_preparation = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
])

# Apply the right preparation steps to the right columns.
preparation = ColumnTransformer([
    ("numbers", numeric_preparation, ["studytime", "absences", "G1", "G2"]),
    ("categories", category_preparation, ["school"]),
])

# Join preparation and regression into one model.
categorical_model = Pipeline([
    ("prepare", preparation),
    ("regression", LinearRegression()),
])

categorical_model.fit(X_train_mixed, y_train_mixed)
mixed_predictions = categorical_model.predict(X_test_mixed)

mixed_rmse = np.sqrt(mean_squared_error(y_test_mixed, mixed_predictions))
mixed_r2 = r2_score(y_test_mixed, mixed_predictions)

print(f"Linear regression + school: RMSE = {mixed_rmse:.2f}, R² = {mixed_r2:.2f}")

# Create a random forest with 200 decision trees.
forest_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)

# Learn from the same training records as the linear model.
forest_model.fit(X_train, y_train)

# Predict final grades for the held-out test records.
forest_predictions = forest_model.predict(X_test)

# Calculate the same scores so we can compare models fairly.
forest_rmse = np.sqrt(mean_squared_error(y_test, forest_predictions))
forest_r2 = r2_score(y_test, forest_predictions)

print(f"Random forest: RMSE = {forest_rmse:.2f}, R² = {forest_r2:.2f}")

# Create five different train/validation splits from the training records.
folds = KFold(n_splits=5, shuffle=True, random_state=42)

# Compare both regression models using the same folds.
models_for_cv = {
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2
    ),
}

for name, model in models_for_cv.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )

    # Scikit-learn returns negative RMSE so that higher scores are better.
    # Change the signs back so RMSE is easier to read.
    rmse_scores = -scores["test_rmse"]
    r2_scores = scores["test_r2"]

    print(
        f"{name} 5-fold CV: "
        f"RMSE = {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}, "
        f"R² = {r2_scores.mean():.2f} ± {r2_scores.std():.2f}"
    )
    
    # Build a path for the saved model file in the project folder.
model_path = Path(__file__).resolve().parent / "student_grade_model.joblib"

# Save the trained random forest.
joblib.dump(forest_model, model_path)

print(f"\nTrained model saved to: {model_path}")
    
    # Put the random forest's feature-importance scores into a readable table.
importance_table = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": forest_model.feature_importances_,
})

# Show the most important features first.
importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom forest feature importance:")
print(importance_table.to_string(index=False))

# Calculate the average absolute prediction error across all test records.
forest_mae = mean_absolute_error(y_test, forest_predictions)
print(f"Random forest: MAE = {forest_mae:.2f} grade points")

# Select one test-set student as a one-row table.
example_student = X_test.iloc[[0]]

# Predict that student's final grade.
predicted_grade = forest_model.predict(example_student)[0]

# Look up the actual final grade for the same test record.
actual_grade = y_test.iloc[0]

print("\nExample prediction:")
print("Inputs:")
print(example_student)
print(f"Predicted final grade: {predicted_grade:.1f}")
print(f"Actual final grade: {actual_grade}")


# Make one hypothetical student's inputs in the same column format as X.
new_student = pd.DataFrame([{
    "studytime": 2,
    "absences": 4,
    "G1": 10,
    "G2": 11,
}])

# Ask the trained random forest for a final-grade prediction.
new_prediction = forest_model.predict(new_student)[0]

print("\nHypothetical student prediction:")
print(new_student)
print(f"Predicted final grade: {new_prediction:.1f}")

# Use only inputs that could be available earlier in the school year.
early_features = ["studytime", "absences"]

# Create a separate random forest for the earlier estimate.
early_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)

# Train it using only the two early inputs.
early_model.fit(X_train[early_features], y_train)

# Predict the same test students using only those inputs.
early_predictions = early_model.predict(X_test[early_features])

# Measure how far these predictions are from the actual final grades.
early_rmse = np.sqrt(mean_squared_error(y_test, early_predictions))
early_r2 = r2_score(y_test, early_predictions)

print("\nEarlier estimate (studytime and absences only):")
print(f"RMSE = {early_rmse:.2f}, R² = {early_r2:.2f}")
print("Later estimate (studytime, absences, G1, G2):")
print("RMSE = 1.83, R² = 0.84")

# Try a third early input: the student's past class failures.
early_features_with_failures = ["studytime", "absences", "failures", "G1", "G2"]

# Select those columns and the final-grade answer from the original table.
X_early = students[early_features_with_failures]
y_early = students["G3"]

# Make a fresh training/test split with the same settings as before.
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_early,
    y_early,
    test_size=0.20,
    random_state=42
)

# Train a random forest using these three early inputs.
early_model_with_failures = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)
early_model_with_failures.fit(X_train_early, y_train_early)

# Predict the test students' final grades and measure the errors.
early_predictions_with_failures = early_model_with_failures.predict(X_test_early)
early_rmse_with_failures = np.sqrt(
    mean_squared_error(y_test_early, early_predictions_with_failures)
)
early_r2_with_failures = r2_score(y_test_early, early_predictions_with_failures)

print("\nEstimate after the second period (G2 available):")
print(f"RMSE = {early_rmse_with_failures:.2f}, R² = {early_r2_with_failures:.2f}")

# Try the later model with past failures included.
late_features_with_failures = [
    "studytime", "absences", "failures", "G1", "G2"
]

# Select the same training students used by the earlier model.
# X_train.index identifies those original student rows.
X_train_late = students.loc[X_train.index, late_features_with_failures]

# Evaluate this version on five parts of the training data.
late_scores = cross_validate(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2
    ),
    X_train_late,
    y_train,
    cv=folds,
    scoring={
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
)

# Convert RMSE back to a positive number and show average scores.
late_rmse_scores = -late_scores["test_rmse"]
late_r2_scores = late_scores["test_r2"]

print("\nLater model with past failures, 5-fold CV:")
print(
    f"RMSE = {late_rmse_scores.mean():.2f} ± {late_rmse_scores.std():.2f}, "
    f"R² = {late_r2_scores.mean():.2f} ± {late_r2_scores.std():.2f}"
)