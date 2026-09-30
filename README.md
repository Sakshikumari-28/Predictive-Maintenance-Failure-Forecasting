# Predictive Maintenance & Machine Failure Forecasting

A machine learning-based predictive maintenance system that analyzes industrial machine operating conditions to identify potential machine failures and classify machines into different risk levels.

The project combines **Python, Pandas, Scikit-learn, Random Forest, and Power BI** to create an end-to-end predictive maintenance pipeline.

---

## 📌 Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and operational losses.

This project aims to analyze machine operating-condition data and predict whether a machine observation is likely to result in a failure.

The system:

- Performs exploratory data analysis (EDA)
- Cleans and preprocesses the dataset
- Trains a Random Forest classification model
- Predicts machine failures
- Calculates failure probability
- Assigns machines to Low, Medium, or High risk categories
- Provides data for an interactive Power BI dashboard

---

## 🎯 Objectives

- Analyze machine operating conditions
- Identify patterns associated with machine failures
- Build a machine failure classification model
- Handle the class imbalance present in failure data
- Generate failure predictions and probabilities
- Classify observations into risk categories
- Visualize machine performance and failure patterns using Power BI

---

## 📊 Dataset

The project uses an enriched version of the AI4I 2020 Predictive Maintenance dataset.

### Dataset size

- **10,000 records**
- **24 input columns**
- **27 columns in the final prediction output**

The dataset contains machine operating conditions, machine information, failure indicators, and additional contextual variables used for analysis and visualization.

### Main ML features

The Random Forest model uses the following five operating-condition features:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

### Target variable

`Machine failure`

Where:

- `0` → No machine failure
- `1` → Machine failure

### Additional enriched variables

The dataset also contains variables such as:

- Machine Age [years]
- Operating Hours
- Maintenance Count
- Hours Since Last Maintenance
- Vibration [mm/s]
- Pressure [bar]
- Power Consumption [kW]
- Ambient Humidity [%]
- Coolant Level [%]

These additional variables provide useful context for analysis and the Power BI dashboard.

---

## 🔎 Exploratory Data Analysis

The EDA pipeline checks:

- Dataset dimensions
- Column names and data types
- Missing values
- Duplicate records
- Unique values
- Target distribution
- Descriptive statistics
- Feature statistics
- Feature means across failure classes
- Correlation with machine failure
- Relationships between operating variables and failure

The dataset contains:

- **0 missing values**
- **0 duplicate rows**

The target distribution contains:

- **9,661 non-failure observations**
- **339 failure observations**

The EDA visualizations are generated automatically and saved under:

```text
data/processed/eda/
