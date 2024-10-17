# Step 1: Importing the required libraries
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Step 2: Creating a simple dataset (X, y)
# Here X is the feature (input), and y is the target (output)
X = np.array([[1], [2], [3], [4], [5]])  # 5 data points
y = np.array([1.5, 2.5, 3.0, 4.5, 5.0])  # corresponding outputs

# Step 3: Training the SVR model
svr_model = SVR(kernel='linear')  # 'linear' kernel is the simplest one
svr_model.fit(X, y)  # Fit the model with data

# Step 4: Making predictions
X_test = np.array([[6], [7], [8]])  # New data points to predict
y_pred = svr_model.predict(X_test)

# Step 5: Evaluating the model (just print the predictions)
print(f"Predictions for {X_test.flatten()} are {y_pred}")

# (Optional) Visualizing the original data and the prediction
plt.scatter(X, y, color='red', label='Original Data')  # Original points
plt.plot(X_test, y_pred, color='blue', label='Predicted Line')  # Predicted line
plt.xlabel('Input Feature (X)')
plt.ylabel('Target Value (y)')
plt.title('Support Vector Regression')
plt.legend()
plt.show()