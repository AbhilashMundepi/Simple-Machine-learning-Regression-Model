from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#  Prepare a simple dataset (Experience vs Salary)
# Experience in years
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
# Salary in thousands
y = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70])

#  Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Predict using the test set
y_pred = model.predict(X_test)

print("Our predicted value is : ", model.predict([[6.5]]))


# Performance of the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Our Mean absolute Error is "f"MAE: {mae}")
print("Our Mean Squared Error is  "f"MSE: {mse}")
print("Our Root Mean Squares Error is  "f"RMSE: {rmse}")
print("Our R Square Score is  "f"R² Score: {r2}")

#  Visualize the result
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (in thousands)')
plt.title('Experience vs Salary Prediction')
plt.legend()
plt.show()




