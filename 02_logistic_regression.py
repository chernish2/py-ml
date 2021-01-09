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


# Cost function aka J(θ) is MSE (Mean Squared Error)
def cost_function(x, y, theta):
    m = len(x)  # number of rows
    sqr_sum = 0
    for i in range(m):
        sqr_sum += pow(hypothesis(x[i], theta) - y[i], 2)
    return sqr_sum / (2 * m)


# Gradient descent algorithm
def gradient_descent(x, y):
    steps_l = []  # list for visualizing data
    n_iter = 500  # number of descent iterations
    visualize_step = round(n_iter / 20)  # we want to have 30 frames in visualization because it looks good
    alpha = 0.05  # learning rate
    m = len(x)  # number of rows
    theta = np.array([0, 0])

    # normalizing feature x_1
    x_norm = x.copy()
    max_x1 = max(np.abs(x[:, 1]))
    x_norm[:, 1] = x_norm[:, 1] / max_x1

    # gradient descent
    for iter in range(n_iter):
        sum_0 = 0
        sum_1 = 0
        for i in range(m):
            y_hypothesis = hypothesis(x_norm[i], theta)
            sum_0 += (y_hypothesis - y[i]) * x_norm[i][0]
            sum_1 += (y_hypothesis - y[i]) * x_norm[i][1]
            if iter % visualize_step == 0:  # add visualization data
                steps_l.append([x[i][1], y_hypothesis, iter])
        new_theta_0 = theta[0] - alpha * sum_0 / m
        new_theta_1 = theta[1] - alpha * sum_1 / m
        theta = [new_theta_0, new_theta_1]
        cost = cost_function(x, y, theta)
        if iter % visualize_step == 0:  # debug output to see what's going on
            print(f'iter={iter}, cost={cost}, theta={theta}')
    print(f'Gradient descent is done with theta_0={theta[0]} and theta_1={theta[1]}')

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
gradient_descent(x, y)
