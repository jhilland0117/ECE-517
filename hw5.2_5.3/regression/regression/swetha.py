import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your data from the CSV file
data = pd.read_csv('data.csv')

# Assuming the last column in your CSV file is the target variable and the rest are features
features = data.iloc[:, :-1]  # Selecting all columns except the last as features
target = data.iloc[:, -1]    # Selecting the last column as the target variable

# Split the data into train and test sets (80% for training, 20% for validation)
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Calculate the trace of the matrix for determining the reasonable interval
trace_X = np.trace(X_train.T @ X_train)
lambda_values = np.logspace(np.log10(0.01 * trace_X), np.log10(10 * trace_X), num=20)

best_lambda = None
min_val_mse = float('inf')

for lambda_val in lambda_values:
    # Train the Ridge regression model
    ridge = Ridge(alpha=lambda_val)
    ridge.fit(X_train, y_train)
    
    # Validate the model
    y_pred_val = ridge.predict(X_val)
    
    # Calculate mean squared error on validation set
    val_mse = mean_squared_error(y_val, y_pred_val)
    
    # Update best lambda if current validation MSE is lower
    if val_mse < min_val_mse:
        min_val_mse = val_mse
        best_lambda = lambda_val
        best_model = ridge

# Use the best model to make predictions on the test set
X_test, y_test = X_val, y_val  # Assuming test set is the validation set
y_pred_test = best_model.predict(X_test)

# Calculate mean squared error on test set
test_mse = mean_squared_error(y_test, y_pred_test)

# Plot real vs predicted data
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title('Real vs Predicted Data')
plt.grid(True)
plt.show()

# Plot validation set square error for each lambda
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, val_mse_list, marker='o', linestyle='-', color='r')
plt.xlabel('Lambda')
plt.ylabel('Validation MSE')
plt.title('Validation Square Error vs Lambda')
plt.xscale('log')
plt.grid(True)
plt.show()

print(f"Optimal Lambda: {best_lambda}")
print(f"Optimal Validation MSE: {min_val_mse}")
print(f"Test MSE with Optimal Lambda: {test_mse}")
