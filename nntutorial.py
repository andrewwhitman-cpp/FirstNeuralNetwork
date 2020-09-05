# following "Beginner Introduction to Neural Networks" on Youtube by giant_neural_network
# given petal data (length, width) and flower colors,
# the code will train, and then be able to predict other flower
# colors based on their petal data

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))


# each value is length, width, type (0 = blue, 1 = red)
data = [[3.0, 1.5, 1],
        [2.0, 1.0, 0],
        [4.0, 1.5, 1],
        [3.0, 1.0, 0],
        [3.5, 0.5, 1],
        [2.0, 0.5, 0],
        [5.5, 1.0, 1],
        [1.0, 1.0, 0]]

# network
#   o    flower type
#  / \   w1, w2, b
# o   o  length, width

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# red flower
mystery_flower1 = [4.5, 1.0]
print("mystery_flower1: (4.5, 1.0)")
# blue flower
mystery_flower2 = [1.0, 2.0]
print("mystery_flower2: (1.0, 2.0)")
print("")

# predict mystery flowers (not trained)
# flower1 is red (close to 1), flower2 is blue (close to 0)
print("predictions before training:")
z = mystery_flower1[0] * w1 + mystery_flower1[1] * w2 + b
pred = sigmoid(z)
print("myster_flower1:", pred)
z = mystery_flower2[0] * w1 + mystery_flower2[1] * w2 + b
pred = sigmoid(z)
print("myster_flower2:", pred)
print("")

# sigmoid and sigmoid prime plots
# X = np.linspace(-5, 5, 100)
# Y = sigmoid(X)
# plt.plot(X, sigmoid(X), c='r')
# plt.plot(X, sigmoid_p(X), c='b')
# plt.show()

# scatter data
plt.axis([0, 6, 0, 6])
plt.grid()
for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 0:
        color = "b"
    plt.scatter(point[0], point[1], c=color)
plt.show()

# training loop
learning_rate = 0.2
costs = []
for i in range(50000):
    random_index = np.random.randint(len(data))
    point = data[random_index]

    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)

    target = point[2]
    cost = np.square(prediction - target)
    costs.append(cost)

    # partial derivatives (manually done, not a good idea for future code)
    dcost_pred = 2 * (prediction - target)
    dpred_dz = sigmoid_p(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dpred_dz

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db
# plt.plot(costs)
# plt.show()

# predict mystery flowers (now trained)
# flower1 is red (close to 1), flower2 is blue (close to 0)
print("predictions after training:")
z = mystery_flower1[0] * w1 + mystery_flower1[1] * w2 + b
pred = sigmoid(z)
print("myster_flower1:", pred)
z = mystery_flower2[0] * w1 + mystery_flower2[1] * w2 + b
pred = sigmoid(z)
print("myster_flower2:", pred)
