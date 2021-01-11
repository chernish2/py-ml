# This is my Python effort on studying the Machine Learning course by Andrew Ng at Coursera
# https://www.coursera.org/learn/machine-learning
#
# Part 1. Linear regression for the straight line y = ax + b

import pandas as pd
import numpy as np
from plotly import express as px
import plotly.graph_objects as go


# Hypothesis function, or prediction function
def hypothesis(x, θ):
    return x[0] * θ[0] + x[1] * θ[1]


# Cost function aka J(θ) is MSE (Mean Squared Error)
def cost_function(x, y, θ):
    m = len(x)  # number of rows
    sqr_sum = 0
    for i in range(m):
        sqr_sum += pow(hypothesis(x[i], θ) - y[i], 2)
    return sqr_sum / (2 * m)


# Gradient descent algorithm
def gradient_descent(x, y, α, normalize=True):
    steps_l = []  # list for visualizing data
    n_iter = 500  # number of descent iterations
    visualize_step = round(n_iter / 20)  # we want to have 20 frames in visualization because it looks good
    m = len(x)  # number of rows
    θ = np.array([0, 0])

    # normalizing feature x_1
    x_norm = x.copy()
    if normalize:
        max_x1 = max(np.abs(x[:, 1]))
        x_norm[:, 1] = x_norm[:, 1] / max_x1

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
        new_θ_0 = θ[0] - α * sum_0 / m
        new_θ_1 = θ[1] - α * sum_1 / m
        θ = [new_θ_0, new_θ_1]
        cost = cost_function(x, y, θ)
        if iter % visualize_step == 0:  # debug output to see what's going on
            print(f'iter={iter}, cost={cost}, θ={θ}')
    print(f'Gradient descent is done with θ_0={θ[0]} and θ_1={θ[1]}')

    # visualizing gradient descent
    df = pd.DataFrame(np.array(steps_l), columns=['x', 'y', 'step'])
    fig = px.scatter(df, x='x', y='y', animation_frame='step')
    fig.add_trace(go.Scatter(x=x[:, 1], y=y, mode='markers'))
    global figure_n
    fig.update_layout(title={'text': f'Figure {figure_n}. α={α}, normalize={normalize}', 'x': 0.5, 'y': 0.9}, font={'size': 18})
    fig.show()
    figure_n += 1


# Generating dataset for our function y = ax + b
# In this case y = 36.78 * x - 150
def generate_dataset():
    a = 36.78
    b = -150
    x = np.arange(-10, 10, step=0.1)
    y_dataset = a * x + b
    noise_ar = np.random.random(y_dataset.shape) * 100  # add some noise to the data
    x_dataset = np.array([[1, x] for x in x])  # adding feature x_0 = 1
    y_dataset = y_dataset + noise_ar
    return x_dataset, y_dataset


x, y = generate_dataset()
α_l = [0.0605, 0.06, 0.05975, 0.05]
figure_n = 1
for α in α_l:
    gradient_descent(x, y, α, False)
α_l = [0.3, 0.03]
for α in α_l:
    gradient_descent(x, y, α)
