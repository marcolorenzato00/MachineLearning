"""Perceptron N-dim"""

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


LEARNING_RATE = 1
df = pd.read_csv("data.csv", sep=";")
x_raw = np.array([row.tolist() for row in df.iloc[:, :-1].to_numpy()])
x_norm = 2 * (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min()) - 1
x = np.hstack([np.ones((x_norm.shape[0], 1)), x_norm])
y = df.iloc[:, -1].to_numpy().tolist()
w = [0 for xi in x[0]]


def sign(a):
    """Return sign of a number"""
    if a >= 0:
        return +1
    else:
        return -1


def scalar_product(a, b):
    """Return the scalar product of 2 vectors"""
    if len(a) != len(b):
        print("Error: vectors of different size, scalar product not computable")
        sys.exit()
    prod = 0
    for ai, bi in zip(a, b):
        prod += ai * bi
    return prod


def compute_w(weight, point, label):
    """Compute new weights after missclassified sample"""
    return [wi + label * LEARNING_RATE * pos for wi, pos in zip(weight, point)]


DONE = False
MAX_STEPS = 1e3
STEPS = 0
while not DONE and STEPS < MAX_STEPS:
    DONE = True
    STEPS += 1
    for xi, yi in zip(x, y):
        p = scalar_product(w, xi)
        if sign(p) != yi:
            w = compute_w(w, xi, yi)
            DONE = False


print(w)

for i, xi in enumerate(x):
    if y[i] == +1:
        plt.scatter(xi[1], xi[2], color="red")
    else:
        plt.scatter(xi[1], xi[2], color="blue")

plt.xlabel("Asse X")
plt.ylabel("Asse Y")
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
x_vals = np.linspace(-1, 1, 100)
y_vals = -(w[0] + w[1] * x_vals) / w[2]
print("Intercept: ", -w[0] / w[2])
print("Slope: ", -w[1] / w[2])
plt.plot(x_vals, y_vals, color="green")

plt.show()
