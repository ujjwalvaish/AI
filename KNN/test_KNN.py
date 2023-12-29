import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from KNN import KNN

##################### CLASSIFICATION ####################
X, y = load_iris(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model creation and training
model = KNN(5)
model.fit(X_train, y_train)
# Testing the model
preds = model.predict(X_test, regression = False)
print(preds)
print(y_test)
accuracy = (np.sum(np.array(preds) == y_test)) / len(y_test)
print("Accuracy: ", accuracy)


##################### REGRESSION ####################


