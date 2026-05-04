import pandas as pd

# -------------------------------
# Load the dataset
# -------------------------------
df = pd.read_csv("smart_healthcare_dataset.csv")


# inspection of Data

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSummary statistics:")
print(df.describe())


# Drop the old health_risk_score


if "health_risk_score" in df.columns:
    print("\nUnique values in health_risk_score:")
    print(df["health_risk_score"].unique())

    df = df.drop(columns=["health_risk_score"])
    print("\nDropped old health_risk_score column.")


# Convert gender to category

df["gender"] = df["gender"].astype("category")


# Create age groups

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 30, 50, 70, 100],
    labels=["young", "mid", "senior", "elderly"],
    right=False
)

print("\nAge group counts:")
print(df["age_group"].value_counts())


# 6. Create exercise risk
# higher exercise = lower risk
# assuming exercise_level is 0 to 3

df["exercise_risk"] = 3 - df["exercise_level"]


# 7. Create a new risk score

df["risk_score"] = (
    df["age"] * 0.2 +
    df["bmi"] * 0.1 +
    df["exercise_risk"] * 5 +
    df["smoking"] * 15 +
    df["alcohol"] * 10 +
    df["blood_pressure"] * 0.15 +
    df["cholesterol"] * 0.15 +
    df["glucose"] * 0.15 +
    df["fatigue"] * 5 +
    df["chest_pain"] * 15 +
    df["dizziness"] * 5 +
    df["heart_disease"] * 25 +
    df["diabetes"] * 20 +
    df["stroke"] * 30
)

print("\nRisk score summary:")
print(df["risk_score"].describe())


# 8. Grouped analysis

print("\nAverage risk score by age group:")
print(df.groupby("age_group", observed=False)["risk_score"].mean())

print("\nAverage risk score by gender:")
print(df.groupby("gender", observed=False)["risk_score"].mean())

print("\nAverage diabetes rate by age group:")
print(df.groupby("age_group", observed=False)["diabetes"].mean())

print("\nAverage stroke rate by age group:")
print(df.groupby("age_group", observed=False)["stroke"].mean())


# 9. create BMI groups

df["bmi_group"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["underweight", "normal", "overweight", "obese"],
    right=False
)

print("\nBMI group counts:")
print(df["bmi_group"].value_counts())

print("\nAverage risk score by BMI group:")
print(df.groupby("bmi_group", observed=False)["risk_score"].mean())


# 10. Save cleaned dataset

df.to_csv("smart_healthcare_cleaned.csv", index=False)
print("\nCleaned dataset saved as smart_healthcare_cleaned.csv")