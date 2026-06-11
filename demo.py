import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["MedHouseVal"] = housing.target

print(df.head())

print("Rows and Columns:")
print(df.shape)

print(df.info())

print(df.isnull().sum())

print("Duplicate Rows:")
print(df.duplicated().sum())

print(df.describe())

plt.figure(figsize=(8,5))

#MedHouseVal

plt.hist(
    df["MedHouseVal"],
    bins=30
)

plt.title("Distribution of House Prices")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")

plt.show()

#Correlation Matrix
corr = df.corr()
print(corr)

#Correlation heatmap

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

#Median Income vs House Price
plt.figure(figsize=(8,5))

plt.scatter(
    df["MedInc"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Income vs House Value")

plt.show() 


#House Age vs House Value
plt.figure(figsize=(8,5))

plt.scatter(
    df["HouseAge"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("House Age")
plt.ylabel("Median House Value")
plt.title("House Age vs House Value")

plt.show()

#Population Distribution
plt.figure(figsize=(8,5))

plt.hist(
    df["Population"],
    bins=40
)

plt.title("Population Distribution")
plt.xlabel("Population")
plt.ylabel("Frequency")

plt.show()

#Boxplots for Outlier Detection
plt.figure(figsize=(10,6))

sns.boxplot(data=df)

plt.xticks(rotation=45)

plt.show()


#1st Testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print(X.shape)
print(y.shape)


# #Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred_lr
})

print(results.head(10))

#R² Score
r2 = r2_score(y_test, pred_lr)

print("R² Score:", r2)

#Mean Absolute Error
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, pred_lr)

print("MAE:", mae)


#Root Mean Squared Error
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, pred_lr)

print("RMSE:", rmse)


#viulasing Predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    pred_lr,
    alpha=0.4
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()

#Coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
})

print(coefficients.sort_values(
    by="Coefficient",
    ascending=False
))

metrics = {
    "R2 Score": r2,
    "MAE": mae,
    "RMSE": rmse
}

print(metrics)



#Random Forest Test
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

r2_rf = r2_score(y_test, pred_rf)

mae_rf = mean_absolute_error(
    y_test,
    pred_rf
)

rmse_rf = root_mean_squared_error(
    y_test,
    pred_rf
)

print("R²:", r2_rf)
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)


comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2": [r2, r2_rf],
    "MAE": [mae, mae_rf],
    "RMSE": [rmse, rmse_rf]
})

print(comparison)


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

import matplotlib.pyplot as plt

importance.plot(
    x="Feature",
    y="Importance",
    kind="bar"
)

plt.title("Feature Importance")
plt.show()

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(rf, file)

with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

sample_prediction = loaded_model.predict(X_test[:1])

print(sample_prediction)

feature_names = X.columns.tolist()

with open("features.pkl", "wb") as file:
    pickle.dump(feature_names, file)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

