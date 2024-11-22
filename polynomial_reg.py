import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Create a sample dataset for employee salary
data = {
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [15000, 20000, 30000, 40000, 60000, 80000, 110000, 150000, 200000, 300000]
}
df = pd.DataFrame(data)

# Independent variable (Level) and dependent variable (Salary)
X = df[['Level']]
y = df['Salary']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a Polynomial Regression Model
poly_features = PolynomialFeatures(degree=3)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict salary using the polynomial regression model
y_pred_train = model.predict(X_poly_train)
y_pred_test = model.predict(X_poly_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"R-squared: {r2}")

# Visualize the Polynomial Regression Results
X_grid = np.arange(min(X['Level']), max(X['Level']) + 0.1, 0.1).reshape(-1, 1)
X_poly_grid = poly_features.transform(X_grid)
y_grid_pred = model.predict(X_poly_grid)

plt.scatter(X['Level'], y, color='blue', label='Actual Salaries')
plt.plot(X_grid, y_grid_pred, color='red', label='Polynomial Regression Line')
plt.title('Polynomial Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
