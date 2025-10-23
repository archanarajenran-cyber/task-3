# Linear Regression - House Price Prediction
# Task 3 - AI & ML Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Load dataset
data = pd.read_csv("Housing.csv")
print("✅ Dataset loaded successfully!\n")
print(data.head())

# 2️⃣ Check for missing values
print("\nMissing values in dataset:\n", data.isnull().sum())

# Convert categorical values (like yes/no, furnishing status) into numeric
data = pd.get_dummies(data, drop_first=True)

# 3️⃣ Define features (X) and target (y)
X = data.drop('price', axis=1)
y = data['price']

# 4️⃣ Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 5️⃣ Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# 6️⃣ Predict
y_pred = model.predict(X_test)

# 7️⃣ Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 8️⃣ Visualize Actual vs Predicted prices
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# 9️⃣ Display model coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n📈 Coefficients:\n", coeff_df)
