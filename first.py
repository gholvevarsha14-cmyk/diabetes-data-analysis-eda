# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load Dataset
df = pd.read_csv("index.csv")

# Display first rows
print("First 5 rows:")
print(df.head())


# Dataset Info
print("\nDataset Info:")
print(df.info())


# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())


# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# EDA : Distribution Plots
plt.figure(figsize=(8, 5))
sns.histplot(df["plasma glucose concentration"], kde=True)
plt.title("Glucose Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["body mass index"], kde=True)
plt.title("BMI Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.show()

# count plot(diabetic vs non-diabetic)
sns.countplot(x="diabetic", data=df)
plt.title("Diabetes Count")
plt.show()

# Boxplots(Outlier detection)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["plasma glucose concentration"])
plt.title("Glucose Boxplot")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["body mass index"])
plt.title("BMI Boxplot")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot
sns.scatterplot(
    x="plasma glucose concentration", y="body mass index", hue="diabetic", data=df
)

plt.title("Glucose vs BMI")
plt.show()

# Feature and Target Selection
X = df[
    [
        "pregnancies",
        "plasma glucose concentration",
        "diastolic blood pressure",
        "triceps skinfold thickness",
        "insulin",
        "body mass index",
        "diabetes pedigree function",
        "age",
    ]
]

y = df["diabetic"]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression Model
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_lr))

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))


# Random Forest Model
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Feature Importance
importances = rf.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)


# Plot Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance")
plt.show()

# Model comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(name, "Accuracy:", accuracy)


# Feature Importance Visualization
importances = rf.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print(importance_df)


# Visualization
sns.barplot(x="Importance", y="Feature", data=importance_df)

plt.title("Feature Importance for Diabetes Prediction")
plt.show()
