import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTrees import DecisionTree

# More info on this data - 
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
breast_cancer_data = datasets.load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target
# 0 - malignant; 1 - benign
print(breast_cancer_data["target"][0], " - " ,breast_cancer_data["target_names"][0])
print(f"Total benign = {np.sum(breast_cancer_data['target'] == 1)}")
print(f"Total malignant = {np.sum(breast_cancer_data['target'] == 0)}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTree(max_depth = 10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy = {np.sum(predictions == y_test) / len(y_test)}")