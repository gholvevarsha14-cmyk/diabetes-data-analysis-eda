# Diabetes Data Analysis (EDA)

## Project Overview
This project performs Exploratory Data Analysis (EDA) on a diabetes dataset to understand the relationship between different health parameters and diabetes outcome.

The analysis helps in identifying patterns, correlations, and trends in the dataset using data visualization and statistical techniques.

## Dataset Features
The dataset contains the following attributes:
- Pregnancies
- Plasma Glucose Concentration
- Diastolic Blood Pressure
- Triceps Skinfold Thickness
- Insulin
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age
- Outcome (Diabetes result)

## Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Steps Performed
1. Data loading and inspection
2. Data cleaning
3. Handling missing values
4. Exploratory Data Analysis (EDA)
5. Data visualization
6. Correlation analysis

## Key Insights
- Glucose level has strong relation with diabetes outcome.
- BMI and Age also show noticeable patterns.
- Visualization helps understand feature distribution.

## Project Structure
- diabetes_analysis.ipynb – Main analysis notebook
- diabetes.csv – Dataset used in the project
- README.md – Project documentation

## Learning Outcome
- Understanding of data preprocessing
- Performing exploratory data analysis
- Creating meaningful data visualizations
- Extracting insights from healthcare data

- ## Results

Four classification models were trained and evaluated on 768 patient records (Pima Indians Diabetes dataset) using an 80/20 train-test split:

| Model | Accuracy |
|---|---|
| Logistic Regression | 75.3% |
| Random Forest | ~75% |
| SVM | 73.4% |
| KNN | 69.5% |

**Logistic Regression achieved the best overall performance**, with the classification report showing 81% precision for non-diabetic cases and 65% precision for diabetic cases.

**Feature Importance (via Random Forest):**

| Feature | Importance |
|---|---|
| Plasma Glucose Concentration | 25.4% |
| Body Mass Index (BMI) | 16.2% |
| Age | 13.9% |
| Diabetes Pedigree Function | 12.5% |

**Key Insight:** Glucose level is by far the strongest predictor of diabetes risk, followed by BMI and age — consistent with established clinical understanding of Type 2 diabetes risk factors.
