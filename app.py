import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv("data/processed/predictions.csv")

with open("models/failure_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Predictive Maintenance Dashboard")
st.write("Machine Failure Prediction System")

st.success("Prediction data loaded successfully!")


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("Dashboard Filters")

selected_risk = st.sidebar.multiselect(
    "Select Risk Band",
    options=df["Risk Band"].dropna().unique(),
    default=df["Risk Band"].dropna().unique()
)

filtered_df = df[
    df["Risk Band"].isin(selected_risk)
].copy()


# ============================================================
# CALCULATE METRICS
# ============================================================

total_observations = len(filtered_df)

predicted_failures = (
    filtered_df["Predicted Failure"] == 1
).sum()

actual_failures = (
    filtered_df["Machine failure"] == 1
).sum()

failure_rate = (
    (actual_failures / total_observations) * 100
    if total_observations > 0
    else 0
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Overview",
    "🔮 Machine Prediction",
    "⚠️ Risk Analysis",
    "📈 Insights"
])


# ============================================================
# TAB 1 — DASHBOARD OVERVIEW
# ============================================================

with tab1:

    st.header("📊 Dashboard Overview")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Observations",
            total_observations
        )

    with col2:
        st.metric(
            "Predicted Failures",
            predicted_failures
        )

    with col3:
        st.metric(
            "Failure Rate",
            f"{failure_rate:.2f}%"
        )

    with col4:
        st.metric(
            "Actual Failures",
            actual_failures
        )

    st.markdown("---")

    # --------------------------------------------------------
    # FAILURE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Failure Distribution")

    failure_distribution = (
        filtered_df["Machine failure"]
        .value_counts()
        .rename(
            index={
                0: "No Failure",
                1: "Failure"
            }
        )
    )

    st.bar_chart(failure_distribution)

    # --------------------------------------------------------
    # PREDICTED VS ACTUAL
    # --------------------------------------------------------

    st.subheader("Predicted vs Actual Failures")

    comparison = pd.DataFrame({
        "Failure Type": [
            "Actual Failures",
            "Predicted Failures"
        ],
        "Count": [
            actual_failures,
            predicted_failures
        ]
    })

    st.bar_chart(
        comparison.set_index("Failure Type")
    )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Risk Band Distribution")

    risk_distribution = (
        filtered_df["Risk Band"]
        .value_counts()
    )

    st.bar_chart(risk_distribution)


# ============================================================
# TAB 2 — MACHINE PREDICTION
# ============================================================

with tab2:

    st.header("🔮 Individual Machine Failure Prediction")

    st.write(
        "Enter the current operating conditions of a machine "
        "to predict whether machine failure is likely."
    )

    col1, col2 = st.columns(2)

    with col1:

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=310.0,
            step=0.1
        )

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=500,
            max_value=5000,
            value=1500,
            step=10
        )

    with col2:

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )

    st.markdown("")

    predict_button = st.button(
        "🔍 Predict Machine Failure",
        use_container_width=True
    )

    if predict_button:

        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame({
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear]
        })

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        # ----------------------------------------------------
        # RISK BAND
        # ----------------------------------------------------

        if probability >= 0.70:
            risk_band = "High"

        elif probability >= 0.30:
            risk_band = "Medium"

        else:
            risk_band = "Low"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            if prediction == 1:
                st.error(
                    "⚠️ MACHINE FAILURE PREDICTED"
                )
            else:
                st.success(
                    "✅ NO MACHINE FAILURE PREDICTED"
                )

        with result_col2:

            st.metric(
                "Failure Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col3:

            st.metric(
                "Risk Band",
                risk_band
            )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("📋 Maintenance Recommendation")

        if risk_band == "High":

            st.error(
                "⚠️ High predicted failure risk. "
                "Preventive maintenance and further "
                "inspection should be considered."
            )

        elif risk_band == "Medium":

            st.warning(
                "⚠️ Medium predicted failure risk. "
                "The machine should be monitored closely."
            )

        else:

            st.success(
                "✅ Low predicted failure risk based "
                "on the current operating conditions."
            )


# ============================================================
# TAB 3 — RISK ANALYSIS
# ============================================================

with tab3:

    st.header("⚠️ Risk Analysis")

    # --------------------------------------------------------
    # HIGH-RISK MACHINES
    # --------------------------------------------------------

    st.subheader("High-Risk Machines")

    high_risk = filtered_df[
        filtered_df["Risk Band"] == "High"
    ].copy()

    high_risk = high_risk.sort_values(
        "Failure Probability",
        ascending=False
    )

    if len(high_risk) > 0:

        st.dataframe(
            high_risk[
                [
                    "UDI",
                    "Type",
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "Failure Probability",
                    "Risk Band"
                ]
            ].head(20),
            use_container_width=True
        )

    else:

        st.info(
            "No high-risk machines found "
            "for the selected risk bands."
        )

    # --------------------------------------------------------
    # FAILURES BY MACHINE TYPE
    # --------------------------------------------------------

    st.subheader("Failures by Machine Type")

    type_failure = (
        filtered_df
        .groupby("Type")["Machine failure"]
        .sum()
        .reset_index()
    )

    type_failure.columns = [
        "Machine Type",
        "Actual Failures"
    ]

    st.bar_chart(
        type_failure.set_index("Machine Type")
    )

    # --------------------------------------------------------
    # FAILURE PROBABILITY
    # --------------------------------------------------------

    st.subheader("Failure Probability Distribution")

    st.line_chart(
        filtered_df[
            "Failure Probability"
        ]
        .sort_values()
        .reset_index(drop=True)
    )


# ============================================================
# TAB 4 — INSIGHTS
# ============================================================

with tab4:

    st.header("📈 Machine Operating Insights")

    # --------------------------------------------------------
    # OPERATING CONDITION SUMMARY
    # --------------------------------------------------------

    st.subheader("Operating Condition Summary")

    avg_air_temp = filtered_df[
        "Air temperature [K]"
    ].mean()

    avg_process_temp = filtered_df[
        "Process temperature [K]"
    ].mean()

    avg_rpm = filtered_df[
        "Rotational speed [rpm]"
    ].mean()

    avg_torque = filtered_df[
        "Torque [Nm]"
    ].mean()

    avg_tool_wear = filtered_df[
        "Tool wear [min]"
    ].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Avg Air Temp",
            f"{avg_air_temp:.2f} K"
        )

    with col2:
        st.metric(
            "Avg Process Temp",
            f"{avg_process_temp:.2f} K"
        )

    with col3:
        st.metric(
            "Avg RPM",
            f"{avg_rpm:.0f}"
        )

    with col4:
        st.metric(
            "Avg Torque",
            f"{avg_torque:.2f} Nm"
        )

    with col5:
        st.metric(
            "Avg Tool Wear",
            f"{avg_tool_wear:.2f} min"
        )

    # --------------------------------------------------------
    # TORQUE VS FAILURE PROBABILITY
    # --------------------------------------------------------

    st.subheader(
        "Failure Probability vs Operating Conditions"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "Torque vs Failure Probability"
        )

        torque_data = (
            filtered_df[
                [
                    "Torque [Nm]",
                    "Failure Probability"
                ]
            ]
            .rename(
                columns={
                    "Torque [Nm]": "Torque"
                }
            )
            .set_index("Torque")
        )

        st.scatter_chart(torque_data)

    # --------------------------------------------------------
    # TOOL WEAR VS FAILURE PROBABILITY
    # --------------------------------------------------------

    with col2:

        st.write(
            "Tool Wear vs Failure Probability"
        )

        tool_wear_data = (
            filtered_df[
                [
                    "Tool wear [min]",
                    "Failure Probability"
                ]
            ]
            .rename(
                columns={
                    "Tool wear [min]": "Tool Wear"
                }
            )
            .set_index("Tool Wear")
        )

        st.scatter_chart(tool_wear_data)

    # --------------------------------------------------------
    # FAILURE STATUS SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Operating Conditions by Failure Status"
    )

    failure_summary = (
        filtered_df
        .groupby("Machine failure")[
            [
                "Torque [Nm]",
                "Tool wear [min]",
                "Rotational speed [rpm]"
            ]
        ]
        .mean()
    )

    failure_summary.index = (
        failure_summary.index.map({
            0: "No Failure",
            1: "Failure"
        })
    )

    st.dataframe(
        failure_summary.round(2),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Predictive Maintenance System | "
    "Machine Failure Forecasting"
)