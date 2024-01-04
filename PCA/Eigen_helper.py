# Explaination here - https://web.physics.utah.edu/~detar/lessons/python/numpy_eigen/node1.html

'''
Covariance - how x changes as y changes; the only difference b/w this and correlation is 
that correlation is constrained between -1 and 1, whereas covariance is not
'''
import numpy as np
import matplotlib.pyplot as plt

# matrix for whic we want to find the eigen vectors
A = np.random.randint(0,10,size = (2,2))

eigenvalues, eigenvectors = np.linalg.eig(A)

# 1st eigen vector 
eigen_1 = eigenvectors[:,0]
# print(f"eigen_1 = {eigen_1}")
lam_1 = eigenvalues[0]
# print(f"lam_1 = {lam_1}")

# Random vector
ran_vec_1 = np.array(np.random.randint(0, 5, size=(2,1)))
ran_vec_2 = np.array(np.random.randint(0, 5, size=(2,1)))

# A*eigen_1 = lam_1*eigen_1
A_eigen_1 = np.dot(A, eigen_1)
lam_1_eigen_1 = np.dot(lam_1, eigen_1)

print(f"A_eigen_1 = {A_eigen_1}")
print(f"lam_1_eigen_1 = {lam_1_eigen_1}")

A_ran_vec_1 = np.dot(A, ran_vec_1)
A_ran_vec_2 = np.dot(A, ran_vec_2)

print(f"A_ran_vec_1 = {A_ran_vec_1} from {ran_vec_1}")
print(f" A_ran_vec_2 = {A_ran_vec_2} from {ran_vec_2}")


# random_vector = np.random.randint(1,5, size=(2,1))
# random_vector_txed = np.dot(A, random_vector)
# print(f"Random vector = {random_vector_txed}")
# print(f"Random vector txed = {random_vector}")
# print(f"Eigen vector = {eigen}")
# print(f"Eigen vector txed = {random_vector}")


# plot

# eigen_txed = np.dot(A, eigen[0])
# V = np.array([[1,1], [-2,2], [4,-7]])
# V = np.array([[random_vector], [eigen], [random_vector_txed], [eigen_txed]])
# origin = np.array([[0, 0],[0, 0]]) # origin point

# plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','o','g'], scale=21)
# plt.show()