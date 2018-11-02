import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import math
from scipy import ndimage
from lr_utils import load_dataset


def sigmoid(z):
    # z = np.array(z)
    s = 1/(1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    #FORWARD PROPAGATION
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.dot(Y, np.log(A.T)) + np.dot(1 - Y, np.log(1-A.T))) / m

    #BACKWARD PROPAGATION
    dz = A - Y
    # dw = np.sum(np.dot(X, dz.T)) / m
    dw = np.dot(X, dz.T) / m
    db = np.sum(dz) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

# print('train_set_x shape:', train_set_x_orig.shape)
# print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print("train_set_y shape: " + str(train_set_y.shape))
# print(train_set_x_flatten)
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
# print(train_set_y.shape)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000, learning_rate=0.005, print_cost=True)
# print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))
# print(train_set_x.shape[0])
# w, b = initialize_with_zeros(train_set_x.shape[0])
# Example of a picture that was wrongly classified.

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
# w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
# print(sigmoid(np.dot(w.T, X) + b).T)
# grads, cost = propagate(w, b, X, Y)
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))
# print("cost = " + str(cost))

# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
# params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

# w = np.array([[0.1124579],[0.23106775]])
# b = -0.3
# X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
# print ("predictions = " + str(predict(w, b, X)))

# print("w = " + str(params["w"]))
# print("b = " + str(params["b"]))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))


# index = 202
# plt.imshow(train_set_x_orig[index])
# plt.show()
