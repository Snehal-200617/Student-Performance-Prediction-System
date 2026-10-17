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
# 2. Professional lavender theme
# ----------------------------

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #F7F5FC;
    color: #302746;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EDE7F8;
    border-right: 1px solid #DED4F0;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #594080 !important;
}

/* General text */
.stApp p,
.stApp label,
.stApp li,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"] p {
    color: #302746;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #44345F;
}

/* Slider labels and values */
[data-testid="stSlider"] label,
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #302746 !important;
}

/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #8064B5;
    border-color: #8064B5;
}

/* Number input */
.stNumberInput input {
    background-color: #FFFFFF;
    color: #302746 !important;
    border-color: #D9CEE9;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #D9CEE9;
}

[data-baseweb="select"] * {
    color: #302746;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #8064B5;
    color: #FFFFFF !important;
    border: 1px solid #8064B5;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #6B4E9C;
    border-color: #6B4E9C;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5DDF2;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #594080 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #302746 !important;
}

/* Forms */
[data-testid="stForm"] {
    background-color: #FCFAFF;
    border: 1px solid #E5DDF2;
    border-radius: 12px;
    padding: 18px;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #E5DDF2;
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #594080 !important;
}

/* Information messages */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #302746 !important;
}

/* Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #E5DDF2;
    border-radius: 10px;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #8064B5;
}

/* Separators */
hr {
    border-color: #E1D7EF;
}
</style>
""", unsafe_allow_html=True)


st.title("📚 Student Performance Prediction")
st.write(
    "Explore student data and estimate a student's final grade "
    "using a machine-learning model."
)


# ----------------------------
# 3. Load dataset and model
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
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

data = load_dataset(DATA_PATH)
model = load_model(MODEL_PATH)

missing = [
    column for column in INPUT_COLUMNS + [TARGET_COLUMN]
    if column not in data.columns
]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()


# ----------------------------
# 4. Navigation
# ----------------------------

section = st.sidebar.radio(
    "Project sections",
    [
        "Overview",
        "Explore the data",
        "Make a prediction",
        "Model results",
    ],
)

st.sidebar.caption("Dataset: UCI Student Performance — Mathematics")


# ----------------------------
# 5. Overview
# ----------------------------

if section == "Overview":

    st.header("Project overview")

    st.write(
        "This project explores whether study time, absences, and previous "
        "grades can help estimate a student's final grade."
    )

    st.write(
        "**G3** is the final grade on a 0–20 scale. "
        "The model uses studytime, absences, G1, and G2."
    )

    first, second, third = st.columns(3)

    first.metric("Student records", len(data))
    second.metric("Dataset columns", len(data.columns))
    third.metric(
        "Average final grade",
        f"{data[TARGET_COLUMN].mean():.1f} / 20",
    )

    st.subheader("Input descriptions")

    st.markdown("""
    - **Study time:** Weekly study-time category from 1 to 4.
    - **Absences:** Recorded school absences.
    - **G1:** First-period grade.
    - **G2:** Second-period grade.
    - **G3:** Final grade estimated by the model.
    """)

    st.info(
        "The model uses G1 and G2, so it is most suitable for estimating "
        "the final grade after these grades are available. Predictions "
        "are estimates, not guaranteed results."
    )

    st.subheader("Tools used")
    st.write("Python, Pandas, Streamlit, and scikit-learn.")


# ----------------------------
# 6. Explore the data
# ----------------------------

elif section == "Explore the data":

    st.header("Explore the student data")

    st.write(
        "These charts summarize patterns in the student dataset. "
        "Relationships do not necessarily indicate causation."
    )

    st.subheader("Distribution of final grades")

    grade_counts = data[TARGET_COLUMN].value_counts().sort_index()
    st.bar_chart(grade_counts)

    st.caption("Each bar represents the number of students receiving a final grade.")

    st.subheader("Average final grade by study-time category")

    average_by_study_time = (
        data.groupby("studytime")[TARGET_COLUMN]
        .mean()
        .rename("Average G3")
    )

    st.bar_chart(average_by_study_time)

    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.subheader("G1 compared with G3")
        st.scatter_chart(data, x="G1", y="G3")
        st.caption("Each point represents one student.")

    with right_chart:
        st.subheader("G2 compared with G3")
        st.scatter_chart(data, x="G2", y="G3")
        st.caption("Each point represents one student.")

    with st.expander("Show a sample of the dataset"):
        st.dataframe(data.head(10), use_container_width=True)

    with st.expander("Show summary statistics"):
        st.dataframe(
            data.describe(include="all").T,
            use_container_width=True,
        )


# ----------------------------
# 7. Make a prediction
# ----------------------------

elif section == "Make a prediction":

    st.header("Prediction lab")

    st.write(
        "Enter the student's details to estimate their final grade."
    )

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

            first_period, second_period = st.columns(2)

            with first_period:
                g1 = st.number_input(
                    "G1 — first-period grade",
                    min_value=0,
                    max_value=20,
                    value=10,
                    step=1,
                )

            with second_period:
                g2 = st.number_input(
                    "G2 — second-period grade",
                    min_value=0,
                    max_value=20,
                    value=10,
                    step=1,
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

            # Convert displayed hours into the dataset's study-time categories.
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

            dataset_average = float(data[TARGET_COLUMN].mean())
            difference = estimate - dataset_average

            st.metric(
                "Estimated final grade",
                f"{estimate:.1f} / 20",
            )

            st.progress(estimate / 20)

            st.caption("Position of the estimate on the 0–20 grade scale.")

            left, right = st.columns(2)

            left.metric(
                "Dataset average G3",
                f"{dataset_average:.1f} / 20",
            )

            right.metric(
                "Difference from average",
                f"{difference:+.1f} points",
            )

            with st.expander("See the profile used for this estimate"):

                readable_profile = student_input.rename(
                    columns={
                        "studytime": "Study-time category",
                        "absences": "Absences",
                        "G1": "First-period grade (G1)",
                        "G2": "Second-period grade (G2)",
                    }
                )

                st.dataframe(
                    readable_profile,
                    hide_index=True,
                    use_container_width=True,
                )

            st.info(
                "This is a model estimate based on patterns in the dataset. "
                "The difference from the dataset average is a comparison, "
                "not a judgment about the student."
            )

        else:

            st.info(
                "Complete the student profile and select "
                "**Create student estimate**. The result will appear here."
            )

            st.markdown("""
            **The model uses:**
            - Study-time category
            - Absences
            - G1, the first-period grade
            - G2, the second-period grade
            """)


# ----------------------------
# 8. Model results
# ----------------------------

elif section == "Model results":

    st.header("How well did the models perform?")

    st.write(
        "The following test-set results compare predictions with grades "
        "that the models did not train on."
    )

    results = pd.DataFrame({
        "Model": [
            "Mean baseline",
            "Linear regression",
            "Random forest",
        ],
        "Test RMSE": [
            4.55,
            2.04,
            1.83,
        ],
        "Test R²": [
            -0.01,
            0.80,
            0.84,
        ],
    })

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Understanding the scores")

    st.markdown("""
    - **RMSE:** Measures the size of prediction errors in grade points.
      Lower values are better.
    - **R²:** Measures how much variation the model explains.
      Values closer to 1 generally indicate a better fit.
    - **Mean baseline:** Predicts the training-set average for every student.
    """)

    st.subheader("Cross-validation check")

    st.write(
        "Five-fold cross-validation evaluates the model using different "
        "training and validation splits."
    )

    st.metric("Average cross-validation RMSE", "1.76 ± 0.42")
    st.metric("Average cross-validation R²", "0.84 ± 0.06")

    st.info(
        "The random forest performed best among the three models in this "
        "experiment. Results may vary with different data splits or settings."
    )

    st.subheader("What I learned")

    st.write(
        "This project helped me understand how machine learning can be used to solve real-world problems in education. It shows how student data can help predict academic performance and identify areas where students may need extra support. Overall, the project aims to make learning more informed, supportive, and effective."
    )