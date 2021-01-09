# This is my Python effort on studying the Machine Learning course by Andrew Ng at Coursera
# https://www.coursera.org/learn/machine-learning
#
# Part 2. Logistic regression

import pandas as pd
import numpy as np
from plotly import express as px
import plotly.graph_objects as go


# Hypothesis function, or prediction function
def hypothesis(x, θ):
    z = x.dot(θ)
    return 1 / (1 + np.power(np.e, -z))


# Cost function aka J(θ)
def cost_function(x, y, θ):
    m = len(x)  # number of rows
    sum = 0
    for i in range(m):
        h = hypothesis(x[i], θ)
        sum += y[i] * np.log(h) + (1 - y[i]) * np.log(1 - h)
    return -sum / m


# Gradient descent algorithm
def gradient_descent(x, y, α):
    steps_l = []  # list for visualizing data
    n_iter = 500  # number of descent iterations
    visualize_step = round(n_iter / 20)  # we want to have 30 frames in visualization because it looks good
    m = len(x)  # number of rows
    θ = np.array([0, 0])

    # normalizing feature x_1
    x_norm = x.copy()
    # max_x1 = max(np.abs(x[:, 1]))
    # x_norm[:, 1] = x_norm[:, 1] / max_x1

    # gradient descent
    for iter in range(n_iter):
        sum_0 = 0
        sum_1 = 0
        for i in range(m):
            y_hypothesis = hypothesis(x_norm[i], θ)
            sum_0 += (y_hypothesis - y[i]) * x_norm[i][0]
            sum_1 += (y_hypothesis - y[i]) * x_norm[i][1]
            if iter % visualize_step == 0:  # add visualization data
                steps_l.append([x[i][1], y_hypothesis, iter])
        new_theta_0 = θ[0] - α * sum_0 / m
        new_theta_1 = θ[1] - α * sum_1 / m
        θ = [new_theta_0, new_theta_1]
        cost = cost_function(x, y, θ)
        if iter % visualize_step == 0:  # debug output to see what's going on
            print(f'iter={iter}, cost={cost}, θ={θ}')
    print(f'Gradient descent is done with theta_0={θ[0]} and theta_1={θ[1]}')

    # visualizing gradient descent
    df = pd.DataFrame(np.array(steps_l), columns=['x', 'y', 'step'])
    fig = px.scatter(df, x='x', y='y', animation_frame='step')
    fig.add_trace(go.Scatter(x=x[:, 1], y=y, mode='markers'))
    fig.show()


def load_dataset():
    dataset = np.genfromtxt('ex2data1.txt', delimiter=',')
    x_dataset = dataset[:, [0, 1]]
    y_dataset = dataset[:, 2]
    return x_dataset, y_dataset


x, y = load_dataset()
λ_l = [0.3, 0.03]
for λ in λ_l:
    gradient_descent(x, y, λ)
