from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Create a random regression problem using sklearn
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# Visualize the data - if X is 1D otherwise won't be able to plot
# fig = plt.figure(figsize=(8,6))
# plt.scatter(X, y, color='b', marker='o', s=30)
# plt.show()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train the Linear Regression algorithm
model = LinearRegression(epochs=1250)
model.train(X_train, y_train)
y_predicted = model.predict(X_test)

# Calculate error metrics
# But this does not make sense for a single model since we do not know
# the magnitudes of X, and Y
# This can however be used to compare two models
rmse = model.calculate_rmse(y_predicted, y_test)
print(f"RMSE = {rmse}")

fig = plt.figure(figsize=(8,6))
# plot X
plt.scatter(X_test, y_test, color='b', marker='o', s=30)
# plot y predicted by our linear regression model
plt.plot(X_test,y_predicted, color='r' )
plt.show()