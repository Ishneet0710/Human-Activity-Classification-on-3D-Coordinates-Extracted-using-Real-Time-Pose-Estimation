import numpy as np
import matplotlib.pyplot as plt

while True:
    try:
        coeff = float(input("Choose Correlation Coefficient: "))
        if -1 <= coeff <= 1:
            break
        else:
            print("Correlation Coefficient must be a number from -1 to 1.")
    except ValueError:
        print("Correlation Coefficient must be a number from -1 to 1.")
        continue

while True:
    try:
        n = int(input("Choose Number of Samples: "))
        if 0 < n <= 100000:
            break
        else:
            print("Number of Samples must be a Positive Integer between 1 and 100000.")
    except ValueError:
        print("Number of Samples must be a Positive Integer between 1 and 100000.")
        continue

cv1 = np.array([[1, coeff], [coeff, 1]]),  # this is a correlation matrix
mu1 = np.array([[1], [2]]),
Z1 = np.random.randn(2, n)
Y1 = np.dot(cv1, Z1) + mu1

cov_mat = np.stack((Y1[0][0], Y1[0][1]), axis=0)
print("Covariance Matrix for rows of Y1:")
print(np.cov(cov_mat))

plt.scatter(Y1[0][0], Y1[0][1])
plt.title("Scatter Plot for randomly generated Y1 data")
plt.xlabel("First Row of Y1")
plt.ylabel("Second Row of Y1")
plt.show()