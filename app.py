from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# 1. Configuration
# ----------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_PATH = PROJECT_FOLDER / "data" / "student-mat.csv"
MODEL_PATH = PROJECT_FOLDER / "student_grade_model.joblib"

INPUT_COLUMNS = ["studytime", "absences", "G1", "G2"]
TARGET_COLUMN = "G3"

st.set_page_config(
    page_title="Student Grade Project",
    page_icon="📚",
    layout="wide",
)

# ----------------------------
# 2. Lavender theme
# ----------------------------

st.markdown("""
<style>
.stApp {
    background-color: #F7F5FC;
    color: #302746;
}

[data-testid="stSidebar"] {
    background-color: #EDE7F8;
    border-right: 1px solid #DED4F0;
}

h1, h2, h3 {
    color: #594080 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #8064B5;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #6B4E9C;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E5DDF2;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stExpander"] {
    background-color: white;
    border: 1px solid #E5DDF2;
    border-radius: 10px;
}

[data-testid="stForm"] {
    background-color: #FCFAFF;
    border: 1px solid #E5DDF2;
    border-radius: 12px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Student Performance Prediction")
st.write(
    "Explore student data and estimate final grades "
    "using a machine-learning model."
)

# ----------------------------
# 3. Load files
# ----------------------------

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path, sep=";")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

data = load_dataset(DATA_PATH)
model = load_model(MODEL_PATH)

missing = [
    col for col in INPUT_COLUMNS + [TARGET_COLUMN]
    if col not in data.columns
]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ----------------------------
# 4. Navigation
# ----------------------------

section = st.sidebar.radio(
    "Project sections",
    ["Overview", "Explore the data", "Make a prediction", "Model results"],
)

st.sidebar.caption("Dataset: UCI Student Performance — Mathematics")

# ----------------------------
# 5. Overview
# ----------------------------

if section == "Overview":
    st.header("Project overview")

    st.write(
        "This project explores how study time, absences, and previous "
        "grades relate to a student's final grade."
    )

    st.write(
        "**G3** is the final grade on a 0–20 scale. "
        "The model uses studytime, absences, G1, and G2."
    )

    a, b, c = st.columns(3)
    a.metric("Student records", len(data))
    b.metric("Dataset columns", len(data.columns))
    c.metric("Average final grade", f"{data['G3'].mean():.1f} / 20")

    st.subheader("Input descriptions")

    st.markdown("""
    - **Study time:** Weekly study-time category.
    - **Absences:** Recorded school absences.
    - **G1:** First-period grade.
    - **G2:** Second-period grade.
    - **G3:** Final grade predicted by the model.
    """)

    st.info("The prediction is an estimate, not a guarantee of the actual grade.")

    st.subheader("Tools used")
    st.write("Python, Pandas, Streamlit, and scikit-learn.")

# ----------------------------
# 6. Explore the data
# ----------------------------

elif section == "Explore the data":
    st.header("Explore the student data")

    st.subheader("Distribution of final grades")
    st.bar_chart(data["G3"].value_counts().sort_index())

    st.subheader("Average grade by study-time category")
    st.bar_chart(data.groupby("studytime")["G3"].mean().rename("Average G3"))

    left, right = st.columns(2)

    with left:
        st.subheader("G1 compared with G3")
        st.scatter_chart(data, x="G1", y="G3")

    with right:
        st.subheader("G2 compared with G3")
        st.scatter_chart(data, x="G2", y="G3")

    with st.expander("Show sample dataset"):
        st.dataframe(data.head(10), use_container_width=True)

    with st.expander("Show summary statistics"):
        st.dataframe(data.describe(include="all").T, use_container_width=True)

# ----------------------------
# 7. Make a prediction
# ----------------------------

elif section == "Make a prediction":
    st.header("Prediction lab")
    st.write("Enter the student's details to estimate the final grade.")

    input_panel, result_panel = st.columns([1, 1.2], gap="large")

    with input_panel:
        st.subheader("Student profile")

        with st.form("student_profile_form"):

            study_hours = st.slider(
                "Study time (0–8 hours/week)",
                min_value=0,
                max_value=8,
                value=5,
                step=1,
            )

            absences = st.number_input(
                "Absences recorded",
                min_value=0,
                max_value=int(data["absences"].max()),
                value=0,
                step=1,
            )

            first, second = st.columns(2)

            with first:
                g1 = st.number_input(
                    "G1 — first-period grade",
                    min_value=0,
                    max_value=20,
                    value=10,
                )

            with second:
                g2 = st.number_input(
                    "G2 — second-period grade",
                    min_value=0,
                    max_value=20,
                    value=10,
                )

            submitted = st.form_submit_button(
                "Create student estimate",
                type="primary",
                use_container_width=True,
            )

        st.caption("Grades are measured on a 0–20 scale.")

    with result_panel:
        st.subheader("Estimate")

        if submitted:

            # Convert hours into the dataset's original categories.
            studytime = min(4, max(1, (study_hours + 1) // 2))

            student_input = pd.DataFrame(
                [{
                    "studytime": studytime,
                    "absences": absences,
                    "G1": g1,
                    "G2": g2,
                }],
                columns=INPUT_COLUMNS,
            )

            estimate = float(model.predict(student_input)[0])
            estimate = max(0.0, min(20.0, estimate))

            average = float(data["G3"].mean())
            difference = estimate - average

            st.metric("Estimated final grade", f"{estimate:.1f} / 20")
            st.progress(estimate / 20)

            left, right = st.columns(2)
            left.metric("Dataset average G3", f"{average:.1f} / 20")
            right.metric("Difference from average", f"{difference:+.1f} points")

            with st.expander("View student profile"):
                st.dataframe(student_input, hide_index=True, use_container_width=True)

            st.info("This is a machine-learning estimate, not a guaranteed result.")

        else:
            st.info("Complete the student profile and select Create student estimate.")

# ----------------------------
# 8. Model results
# ----------------------------

elif section == "Model results":
    st.header("How well did the models perform?")

    results = pd.DataFrame({
        "Model": ["Mean baseline", "Linear regression", "Random forest"],
        "Test RMSE": [4.55, 2.04, 1.83],
        "Test R²": [-0.01, 0.80, 0.84],
    })

    st.dataframe(results, hide_index=True, use_container_width=True)

    st.subheader("Understanding the scores")

    st.markdown("""
    - **RMSE:** Measures prediction error. Lower is better.
    - **R²:** Measures how much variation the model explains. Higher is generally better.
    - **Mean baseline:** Predicts the training-set average for every student.
    """)

    st.subheader("Cross-validation check")
    st.metric("Average cross-validation RMSE", "1.76 ± 0.42")
    st.metric("Average cross-validation R²", "0.84 ± 0.06")

    st.info("The random forest performed best among the three models in this experiment.")

    st.subheader("What I learned")
    st.write(
        "Describe what you learned about the dataset, model evaluation, "
        "and the limitations of grade prediction."
    )