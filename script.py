# Implementation of Gradient Descent for Linear Regression
# y = wx + b
# loss = (y- yhat)^2/n

#Steps
#initialize the parameters w and b
#calculate the gradient descent function
#iteratively make updates

import numpy as np

x = np.random.randn(10,1)
y = 2*x + np.random.rand()

# parameters
w = 0.0
b = 0.0

# hyperparameters
learning_rate = 0.01

# creating a function for gradient descent
def gradient_descent(x,y,w,b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    for xi, yi in zip(x,y):
        dldw += -2*xi*(yi - (w*xi + b))
        dldb += -2*(yi - (w*xi + b))

    # makign an update on the w  and b parameters
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb

    return w,b

# making updates
for epoch  in range(400):
    w, b = gradient_descent(x,y,w,b,learning_rate)
    yhat = w * x + b
    loss = np.divide(np.sum((y - yhat)**2, axis=0), x.shape[0])
    print(f"{epoch} loss is {loss}, parameters w:{w}, b:{b}")
