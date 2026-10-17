from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ----------------------------
# 1. Find the project files
# ----------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

DATA_PATH = PROJECT_FOLDER / "data" / "student-mat.csv"
MODEL_PATH = PROJECT_FOLDER / "student_grade_model.joblib"

INPUT_COLUMNS = ["studytime", "absences", "G1", "G2"]
TARGET_COLUMN = "G3"


# ----------------------------
# 2. Set up the page
# ----------------------------

st.set_page_config(
    page_title="Student Grade Project",
    page_icon="📚",
    layout="wide",
)

# ----------------------------
# 3. Professional lavender theme
# ----------------------------

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #F7F5FC;
        color: #302746;
    }

    /* Main content area */
    [data-testid="stMain"] {
        background-color: #F7F5FC;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #EDE7F8;
        border-right: 1px solid #DED4F0;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #594080 !important;
    }

    /* General text */
    .stApp p,
    .stApp label,
    .stApp .stMarkdown {
        color: #302746;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #44345F;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #8064B5;
        color: #FFFFFF;
        border: 1px solid #8064B5;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }

    /* Button hover */
    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #6B4E9C;
        color: #FFFFFF;
        border-color: #6B4E9C;
    }

    /* Keyboard focus */
    .stButton > button:focus,
    .stFormSubmitButton > button:focus {
        color: #FFFFFF;
        border-color: #594080;
        box-shadow: 0 0 0 2px #D7C8F0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5DDF2;
        border-radius: 12px;
        padding: 18px;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        color: #66577D;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #594080;
    }

    /* Input fields */
    .stNumberInput input {
        background-color: #FFFFFF;
        color: #302746;
        border-radius: 8px;
    }

    /* Select boxes */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #D9CEE9;
        border-radius: 8px;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E5DDF2;
        border-radius: 10px;
    }

    /* Expandable section headings */
    [data-testid="stExpander"] summary {
        color: #594080;
    }

    /* Information and status messages */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Data tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #E5DDF2;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Form containers */
    [data-testid="stForm"] {
        background-color: #FCFAFF;
        border: 1px solid #E5DDF2;
        border-radius: 12px;
        padding: 20px;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #8064B5;
    }

    /* Sidebar radio selection */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #44345F;
    }

    /* Horizontal separators */
    hr {
        border-color: #E1D7EF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Student Performance Prediction")
st.write(
    "This project explores student grade data and uses a machine-learning "
    "model to estimate a student's final grade."
)


# ----------------------------
# 4. Load the CSV and model
# ----------------------------

@st.cache_data
def load_dataset(file_path):
    """Read the semicolon-separated CSV and return it as a table."""
    return pd.read_csv(file_path, sep=";")


@st.cache_resource
def load_model(file_path):
    """Load the trained machine-learning model from disk."""
    return joblib.load(file_path)


if not DATA_PATH.exists():
    st.error(
        f"Could not find the dataset here:\n\n{DATA_PATH}\n\n"
        "Check that student-mat.csv is inside your project's data folder."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        f"Could not find the trained model here:\n\n{MODEL_PATH}\n\n"
        "Check that student_grade_model.joblib is beside app.py."
    )
    st.stop()

data = load_dataset(DATA_PATH)
model = load_model(MODEL_PATH)

needed_columns = INPUT_COLUMNS + [TARGET_COLUMN]
missing_columns = [
    column for column in needed_columns if column not in data.columns
]

if missing_columns:
    st.error(f"The CSV is missing these required columns: {missing_columns}")
    st.stop()


# ----------------------------
# 5. Navigation
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
# 6. Overview section
# ----------------------------

if section == "Overview":
    st.header("Project overview")

    st.write(
        "The aim of this project is to explore whether information about "
        "a student's study time, absences, and earlier grades can help "
        "estimate their final grade."
    )

    st.write(
        "The target is **G3**, the final grade on a 0–20 scale. "
        "The prediction model uses **studytime**, **absences**, **G1**, "
        "and **G2** as its inputs."
    )

    first, second, third = st.columns(3)

    with first:
        st.metric("Student records", len(data))

    with second:
        st.metric("Dataset columns", len(data.columns))

    with third:
        st.metric(
            "Average final grade",
            f"{data[TARGET_COLUMN].mean():.1f} / 20",
        )

    st.subheader("What the input columns mean")

    st.markdown(
        """
        - **Study time:** a category from 1 to 4. In this dataset, 1 means
          under 2 hours per week and 4 means over 10 hours per week.
        - **Absences:** the recorded number of school absences.
        - **G1:** first-period grade.
        - **G2:** second-period grade.
        - **G3:** final grade, which the model estimates.
        """
    )

    st.info(
        "This model uses G1 and G2, so it is most suitable for an estimate "
        "after those grades are available. It does not guarantee a student's "
        "actual final result."
    )

    st.subheader("Tools used")
    st.write("Python, Pandas, Streamlit, and a trained scikit-learn model.")


# ----------------------------
# 7. Explore the data section
# ----------------------------

elif section == "Explore the data":
    st.header("Explore the student data")

    st.write(
        "These charts summarize patterns in the dataset. A pattern or "
        "relationship does not prove that one factor causes another."
    )

    st.subheader("How final grades are distributed")
    grade_counts = data[TARGET_COLUMN].value_counts().sort_index()
    st.bar_chart(grade_counts)

    st.caption(
        "Each bar shows the number of students who received that G3 grade."
    )

    st.subheader("Average final grade by study-time category")
    average_by_study_time = (
        data.groupby("studytime")[TARGET_COLUMN]
        .mean()
        .rename("Average G3")
    )
    st.bar_chart(average_by_study_time)

    st.caption(
        "Study-time categories are coded 1–4. This chart compares group averages; "
        "it does not show that study time alone caused the grades."
    )

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
# 8. Make a prediction section
# ----------------------------

elif section == "Make a prediction":
    st.header("Prediction lab")

    st.write(
        "Build a student profile using the four details the model was trained "
        "to use. Submit the profile to see an estimated final grade."
    )

    input_panel, result_panel = st.columns([1, 1.2], gap="large")

    with input_panel:
        st.subheader("Student profile")

        study_time_labels = {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
        }

        with st.form("student_profile_form"):
            studytime = st.selectbox(
                "Weekly study time",
                options=[1, 2, 3, 4],
                format_func=lambda value: (
                    f"Category {value}: {study_time_labels[value]}"
                ),
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

        st.caption(
            "Grades use a 0–20 scale. Study-time categories follow the "
            "codes in the dataset."
        )

    with result_panel:
        st.subheader("Estimate")

        if submitted:
            student_input = pd.DataFrame(
                [
                    {
                        "studytime": studytime,
                        "absences": absences,
                        "G1": g1,
                        "G2": g2,
                    }
                ],
                columns=INPUT_COLUMNS,
            )

            estimate = float(model.predict(student_input)[0])
            estimate = max(0.0, min(20.0, estimate))

            dataset_average = float(data[TARGET_COLUMN].mean())
            difference = estimate - dataset_average

            st.metric(
                label="Estimated final grade",
                value=f"{estimate:.1f} / 20",
            )

            st.progress(estimate / 20)

            st.caption("Position of the estimate on the 0–20 grade scale")

            comparison_left, comparison_right = st.columns(2)

            with comparison_left:
                st.metric(
                    "Dataset average G3",
                    f"{dataset_average:.1f} / 20",
                )

            with comparison_right:
                st.metric(
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
                "Complete the student profile and select **Create student "
                "estimate**. The result will appear here."
            )

            st.markdown(
                """
                **The model uses:**
                - Study-time category
                - Absences
                - G1, the first-period grade
                - G2, the second-period grade
                """
            )


# ----------------------------
# 9. Model results section
# ----------------------------

elif section == "Model results":
    st.header("How well did the models perform?")

    st.write(
        "The values below are from the model evaluation you ran earlier. "
        "The test-set results compare predictions with grades the models "
        "did not train on."
    )

    results = pd.DataFrame(
        {
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
        }
    )

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("How to read these scores")

    st.markdown(
        """
        - **RMSE** is the typical size of prediction errors, measured in grade
          points. Lower is better. The random forest's test RMSE was **1.83**.
        - **R²** compares the model with a simple average-based prediction.
          A value closer to 1 means the model explained more of the variation
          in the test grades. The random forest's test R² was **0.84**.
        - The **mean baseline** is a simple reference model that predicts the
          training-set average for every student.
        """
    )

    st.subheader("Cross-validation check")

    st.write(
        "In 5-fold cross-validation, the training data is split into five parts. "
        "The model is fitted and checked several times using different parts. "
        "Your random forest results were:"
    )

    st.metric("Average cross-validation RMSE", "1.76 ± 0.42")
    st.metric("Average cross-validation R²", "0.84 ± 0.06")

    st.info(
        "Your evaluation suggests the random forest performed best among these "
        "three models in this experiment. Scores can change if the data split "
        "or model settings change."
    )

    st.subheader("What I learned")

    st.write(
        "Write this part in your own words before showing the project. For example, "
        "describe one thing you learned about the dataset, one thing you learned "
        "about model evaluation, and one limitation of your prediction."
    )