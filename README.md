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
🤖 Machine Learning
Algorithm

The project uses:

Random Forest Classifier

The model is trained using the five selected operating-condition features.

To address the imbalanced target distribution, the model uses:

class_weight="balanced"
Train/Test Split
Training set: 80%
Test set: 20%
Random state: 42
Stratified split
📈 Model Performance

The current model achieved the following results on the test set:

Metric	Score
Accuracy	98.15%
Precision	73.13%
Recall	72.06%
F1 Score	72.59%
Confusion Matrix
                Predicted
                0     1

Actual  0     1914   18
        1       19   49

The model correctly identified 49 of the 68 failure cases in the test set.

⚠️ Failure Probability

In addition to binary failure prediction, the system generates a failure probability using:

model.predict_proba(X)[:, 1]

This allows the system to provide a probability-based assessment rather than only a binary prediction.

🚦 Risk Classification

The predicted failure probability is converted into three risk categories:

Failure Probability	Risk Band
< 30%	Low
30% – 60%	Medium
> 60%	High

The latest generated prediction dataset contains:

Risk Band	Records
Low	9,530
Medium	146
High	324
🏗️ Project Structure
Predictive-Maintenance-Failure-Forecasting/
│
├── data/
│   └── raw/
│       └── ai4i2020.csv
│
├── src/
│   ├── data_processing/
│   │   ├── eda.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   └── train_model.py
│   │
│   └── prediction/
│       └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md

Generated files such as processed datasets, prediction outputs, EDA images, and trained model files are excluded from Git using .gitignore.

⚙️ Installation

Clone the repository:

git clone https://github.com/Sakshikumari-28/Predictive-Maintenance-Failure-Forecasting.git

Navigate into the project:

cd Predictive-Maintenance-Failure-Forecasting

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project
1. Run EDA
python src/data_processing/eda.py

This analyzes the dataset and generates EDA visualizations.

2. Preprocess the dataset
python src/data_processing/preprocess.py

This creates:

data/processed/cleaned_data.csv
3. Train the model
python src/models/train_model.py

This trains the Random Forest model and saves the trained model locally.

4. Generate predictions
python src/prediction/predict.py

This generates:

data/processed/predictions.csv

The output contains:

Actual machine failure
Predicted failure
Failure probability
Risk band
Machine operating conditions
Enriched machine information
📊 Power BI Dashboard

The prediction output is designed to be used as the data source for an interactive Power BI dashboard.

The planned dashboard includes:

Total machine observations
Actual failures
Predicted failures
Failure rate
Failure distribution
Actual vs predicted failures
Failure analysis by machine type
Tool wear analysis
Torque and rotational-speed analysis
Temperature analysis
Vibration analysis
Power consumption analysis
Risk distribution
High-risk machine identification
⚠️ Limitations
The model predicts machine failure classification rather than remaining useful life (RUL).
The dataset is an AI4I predictive-maintenance dataset with additional enriched variables and should not be treated as direct real-world production data.
The current model uses five operating-condition features for prediction.
The failure dataset is highly imbalanced.
Model performance has been evaluated using a held-out test set from the available dataset.
🔮 Future Improvements

Potential future improvements include:

Hyperparameter optimization
Comparison with XGBoost, Gradient Boosting, and other classifiers
Cross-validation
Threshold optimization for failure detection
Explainable AI using SHAP
Remaining Useful Life prediction
Real-time machine monitoring
Automated maintenance alerts
Integration with IoT sensor streams
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Pickle
Power BI
Git & GitHub
👩‍💻 Author

Sakshi Kumari

Computer Science & Engineering – Artificial Intelligence & Machine Learning


## 2. GitHub "About" section

For the short **About** description on GitHub, use:

> **Machine learning-based predictive maintenance system that forecasts machine failures, estimates failure probability, and classifies machine risk using Random Forest and Power BI.**

For the repository **topics**, I'd use:

```text
predictive-maintenance
machine-learning
random-forest
python
scikit-learn
power-bi
data-analysis
machine-failure-prediction
predictive-analytics
