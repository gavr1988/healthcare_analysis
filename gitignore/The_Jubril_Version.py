import pandas as pd
import numpy as np

df = pd.read_csv('smart_healthcare_dataset.csv')

# Transformation: Drop constant column
if 'health_risk_score' in df.columns:
    df = df.drop(columns=['health_risk_score'])

# Transformation: Map binary 0/1 to No/Yes for readable visual legends
binary_cols = ['smoking', 'alcohol', 'fatigue', 'chest_pain', 'dizziness', 'heart_disease', 'diabetes', 'stroke']
for col in binary_cols:
    df[col] = df[col].map({0: 'No', 1: 'Yes'})

# Transformation: Map exercise level
df['exercise_level'] = df['exercise_level'].map({0: 'Low', 1: 'Moderate', 2: 'High'})

# Transformation: Create Age Groups for easier bar charts/demographics
bins_age = [17, 35, 50, 65, 100]
labels_age = ['18-35', '36-50', '51-65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins_age, labels=labels_age)

# Transformation: BMI Categories (WHO Standard)
bins_bmi = [0, 18.5, 24.9, 29.9, 100]
labels_bmi = ['Underweight', 'Normal Weight', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi)

# Save the transformed dataset
output_file = 'smart_healthcare_dataset_transformed.csv'
df.to_csv(output_file, index=False)
print(f"File saved to {output_file}")
print(df.head())