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
    result = 1 / (1 + np.power(np.e, -z))
    # if result == 1 or result == 0:
    #     print(f'x={x}, θ={θ}, z={z}, result={result}')
    #     exit()
    return result


# Cost function aka J(θ)
def cost_function(x, y, θ):
    m = len(x)  # number of rows
    sum = 0
    for i in range(m):
        h = hypothesis(x[i], θ)
        x_i = x[i]
        y_i = y[i]
        if y[i] == 1:
            sum += np.log(h)
        else:
            sum += np.log(1 - h)
    return -sum / m


# Gradient descent algorithm
def gradient_descent(x, y, α):
    steps_l = []  # list for visualizing data
    n_iter = 1500  # number of descent iterations
    visualize_step = round(n_iter / 20)  # we want to have 20 frames in visualization because it looks good
    m = len(x)  # number of rows
    # θ = np.array([0.1, 0.1, 0.1])
    θ = np.array([0, 0, 0])

    # normalizing feature x_1
    x_norm = x.copy()
    # max_x1 = max(np.abs(x[:, 1]))
    # x_norm[:, 1] = x_norm[:, 1] / max_x1

    # gradient descent
    for iter in range(n_iter):
        sum_0 = 0
        sum_1 = 0
        sum_2 = 0
        for i in range(m):
            y_hypothesis = hypothesis(x_norm[i], θ)
            sum_0 += (y_hypothesis - y[i]) * x_norm[i][0]
            sum_1 += (y_hypothesis - y[i]) * x_norm[i][1]
            sum_2 += (y_hypothesis - y[i]) * x_norm[i][2]
        new_θ_0 = θ[0] - α * sum_0 / m
        new_θ_1 = θ[1] - α * sum_1 / m
        new_θ_2 = θ[2] - α * sum_2 / m
        θ = [new_θ_0, new_θ_1, new_θ_2]
        cost = cost_function(x, y, θ)
        if iter % visualize_step == 0:  # debug output to see what's going on
            print(f'iter={iter}, cost={cost}, θ={θ}')
            # if iter % visualize_step == 0:  # add visualization data
            # x1_min = np.min(x[:, 1])
            # x1_max = np.max(x[:, 1])
            for x1 in [60, 70]:
                x2 = -(θ[0] + θ[1] * x_norm[i, 1]) / θ[2]
                steps_l.append([x1, x2, iter])

    print(f'Gradient descent is done with theta_0={θ[0]} and theta_1={θ[1]}')

    # visualizing gradient descent
    df = pd.DataFrame(np.array(steps_l), columns=['x1', 'x2', 'step'])
    fig = px.scatter(df, x='x1', y='x2', animation_frame='step').update_traces(mode='lines+markers')
    positive_l = []
    negative_l = []
    for i in range(m):
        if y[i] == 1:
            positive_l.append(x[i, [1, 2]].tolist())
        else:
            negative_l.append(x[i, [1, 2]].tolist())
    positive_a = np.array(positive_l)
    negative_a = np.array(negative_l)
    fig.add_trace(go.Scatter(x=positive_a[:, 0], y=positive_a[:, 1], mode='markers', name='positive'))
    fig.add_trace(go.Scatter(x=negative_a[:, 0], y=negative_a[:, 1], mode='markers', name='negative'))
    square_size = 960
    axis_range = [30, 100]
    fig.update_layout(autosize=False, width=square_size, height=square_size, xaxis_range=axis_range, yaxis_range=axis_range)
    fig.update_xaxes(title='x1')
    fig.update_yaxes(title='x2')
    # fig.show()


def load_dataset():
    dataset = np.genfromtxt('ex2data1.txt', delimiter=',')
    x = dataset[:, [0, 1]]
    x0_column = np.ones((len(x), 1))  # adding feature x_0 = 1
    x_dataset = np.append(x0_column, x, axis=1)
    y_dataset = dataset[:, 2]
    return x_dataset, y_dataset


x, y = load_dataset()
α_l = [0.02]
for α in α_l:
    gradient_descent(x, y, α)

# t=[0.1,0.1,0.1]
# print(cost_function(x, y, t))
# print(hypothesis(x[0],t))
