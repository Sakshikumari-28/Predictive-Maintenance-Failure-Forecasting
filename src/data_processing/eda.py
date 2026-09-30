import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/raw/ai4i2020.csv")

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
for column in df.columns:
    print("-", column)


# ============================================================
# 2. DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print(f"\nTotal missing values: {missing_values.sum()}")


# ============================================================
# 4. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# ============================================================
# 5. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

print(df.nunique().sort_values())


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df["Machine failure"].value_counts().sort_index()
target_percentages = df["Machine failure"].value_counts(
    normalize=True
).sort_index() * 100

print("\nCounts:")
print(target_counts)

print("\nPercentages:")
print(target_percentages)


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe().T)


# ============================================================
# 8. ORIGINAL FIVE ML FEATURES
# ============================================================

ml_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("\n" + "=" * 60)
print("ML FEATURE STATISTICS")
print("=" * 60)

print(df[ml_features].describe().T)


# ============================================================
# 9. FAILURE VS NON-FAILURE FEATURE MEANS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE MEANS BY MACHINE FAILURE")
print("=" * 60)

failure_means = df.groupby("Machine failure")[ml_features].mean().T

print(failure_means)


# ============================================================
# 10. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION WITH MACHINE FAILURE")
print("=" * 60)

numeric_df = df.select_dtypes(include="number")

target_correlation = (
    numeric_df.corr()["Machine failure"]
    .drop("Machine failure")
    .sort_values(key=abs, ascending=False)
)

print(target_correlation)


# ============================================================
# 11. ENRICHED FEATURE CORRELATIONS
# ============================================================

enriched_features = [
    "Machine Age [years]",
    "Operating Hours",
    "Maintenance Count",
    "Hours Since Last Maintenance",
    "Vibration [mm/s]",
    "Pressure [bar]",
    "Power Consumption [kW]",
    "Ambient Humidity [%]",
    "Coolant Level [%]"
]

available_enriched = [
    column for column in enriched_features
    if column in df.columns
]

print("\n" + "=" * 60)
print("ENRICHED FEATURE CORRELATION WITH FAILURE")
print("=" * 60)

enriched_correlation = (
    df[available_enriched + ["Machine failure"]]
    .corr()["Machine failure"]
    .drop("Machine failure")
    .sort_values(key=abs, ascending=False)
)

print(enriched_correlation)


# ============================================================
# 12. CREATE EDA OUTPUT DIRECTORY
# ============================================================

output_dir = Path("data/processed/eda")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 13. TARGET DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(7, 5))

target_counts.plot(
    kind="bar"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Observations")

plt.xticks(
    [0, 1],
    ["No Failure", "Failure"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    output_dir / "target_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. DISTRIBUTION OF ML FEATURES
# ============================================================

for feature in ml_features:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df[feature],
        bins=30
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = (
        feature
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        output_dir / f"{filename}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 15. FAILURE VS TOOL WEAR
# ============================================================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["Machine failure"] == 0, "Tool wear [min]"],
        df.loc[df["Machine failure"] == 1, "Tool wear [min]"]
    ],
    tick_labels=["No Failure", "Failure"]
)

plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Tool Wear [min]")

plt.tight_layout()

plt.savefig(
    output_dir / "tool_wear_vs_failure.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. TORQUE VS FAILURE
# ============================================================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["Machine failure"] == 0, "Torque [Nm]"],
        df.loc[df["Machine failure"] == 1, "Torque [Nm]"]
    ],
    tick_labels=["No Failure", "Failure"]
)

plt.title("Torque vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Torque [Nm]")

plt.tight_layout()

plt.savefig(
    output_dir / "torque_vs_failure.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. ROTATIONAL SPEED VS FAILURE
# ============================================================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["Machine failure"] == 0, "Rotational speed [rpm]"],
        df.loc[df["Machine failure"] == 1, "Rotational speed [rpm]"]
    ],
    tick_labels=["No Failure", "Failure"]
)

plt.title("Rotational Speed vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Rotational Speed [rpm]")

plt.tight_layout()

plt.savefig(
    output_dir / "rotational_speed_vs_failure.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. TEMPERATURE VS FAILURE
# ============================================================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["Machine failure"] == 0, "Air temperature [K]"],
        df.loc[df["Machine failure"] == 1, "Air temperature [K]"]
    ],
    tick_labels=["No Failure", "Failure"]
)

plt.title("Air Temperature vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Air Temperature [K]")

plt.tight_layout()

plt.savefig(
    output_dir / "air_temperature_vs_failure.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. VIBRATION VS FAILURE
# ============================================================

if "Vibration [mm/s]" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [
            df.loc[df["Machine failure"] == 0, "Vibration [mm/s]"],
            df.loc[df["Machine failure"] == 1, "Vibration [mm/s]"]
        ],
        tick_labels=["No Failure", "Failure"]
    )

    plt.title("Vibration vs Machine Failure")
    plt.xlabel("Machine Failure")
    plt.ylabel("Vibration [mm/s]")

    plt.tight_layout()

    plt.savefig(
        output_dir / "vibration_vs_failure.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print(f"EDA plots saved to: {output_dir}")
