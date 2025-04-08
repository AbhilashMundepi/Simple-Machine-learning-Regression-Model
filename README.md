# 📊 Experience vs Salary Prediction using Linear Regression

This is a simple Machine Learning project that demonstrates **Linear Regression** to predict **Salary (in thousands)** based on **Years of Experience**.

---

## 📌 Project Highlights

- ✅ Trains a linear regression model using scikit-learn.
- 🧪 Splits dataset into training and testing sets.
- 📈 Evaluates the model using metrics like MAE, MSE, RMSE, and R² Score.
- 📉 Visualizes actual vs predicted data using matplotlib.

---

## 🧠 Dataset

The dataset is hardcoded in the script as NumPy arrays:

```python
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70])
