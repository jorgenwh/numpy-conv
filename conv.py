import torch
import numpy as np
import matplotlib.pyplot as plt

def produce_dataset():
    x = np.random.randn(3, 3).astype(np.float32)
    k = np.random.randn(2, 2).astype(np.float32)
    t = np.random.randn(2, 2).astype(np.float32)
    return x, k, t

def conv_forward(x, k):
    k_width = k.shape[0]
    k_height = k.shape[1]
    y_width = x.shape[0] - k.shape[0] + 1
    y_height = x.shape[1] - k.shape[1] + 1
    y = np.zeros((y_width, y_height), dtype=np.float32)
    for y_h in range(y_height):
        for y_w in range(y_width):
            for k_h in range(k_height):
                for k_w in range(k_width):
                    y[y_w, y_h] += x[y_w+k_w, y_h+k_h] * k[k_w, k_h]
            y[y_w, y_h] /= (k_width * k_height)
    return y

def relu_forward(x):
    return np.maximum(x, 0)

def mse_forward(y, t):
    return np.mean((y - t)**2)

def mse_backward(y, t):
    return 2*(y - t)

def relu_backward(dL):
    return np.where(dL > 0, 1, 0)

def conv_backward(dy, x, k):
    dk = np.zeros_like(k)

    # dk
    k_width = k.shape[0]
    k_height = k.shape[1]
    y_width = x.shape[0] - k.shape[0] + 1
    y_height = x.shape[1] - k.shape[1] + 1
    for y_h in range(y_height):
        for y_w in range(y_width):
            for k_h in range(k_height):
                for k_w in range(k_width):
                    dk[k_w, k_h] += dy[y_w, y_h] * x[y_w+k_w, y_h+k_h]
    dk /= (k_width * k_height)

    return dk

if __name__ == "__main__":
    x, k, t = produce_dataset()
    for _ in range(100):
        # forward
        z = conv_forward(x, k)
        y = relu_forward(z)
        loss = mse_forward(y, t)
        if _ % 2 == 0:
            print(loss)

        # backward
        dL = mse_backward(y, t)
        dy = dL * relu_backward(dL)
        dk = conv_backward(dy, x, k)

        # update
        k += -0.01 * dk
