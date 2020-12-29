import pandas as pd
import calendar
import numpy as np
from plotly import express as px
import plotly.graph_objects as go
from plotly.graph_objs.layout import yaxis
from plotly.validators.carpet.aaxis import _ticktext
from sklearn import linear_model

# x_l = np.array([1, 2, 3, 4, 5, 6, 7])
# x = x_l.reshape(-1, 1)
# # y = np.array([6, 9, 11, 13, 15, 19])
# y = np.array([5, -5, 5, -5, 5, -5, 5])
# reg = linear_model.LinearRegression()
# # reg = linear_model.ElasticNet()
# reg.fit(x, y)
#
# pr_x_l = np.array([-1, 0, 7, 10])
# pr_x = pr_x_l.reshape(-1, 1)
# pr_y = reg.predict(pr_x)
#
# fig = px.scatter(x=x_l, y=y)
# fig.add_trace(go.Scatter(x=pr_x_l, y=pr_y, mode='lines+markers', line=go.scatter.Line(color='red')))
# # fig.add_trace(go.Scatter(x=[1,2], y=[1,2], mode='lines',line=go.scatter.Line(color='red')))
# fig.show()
#
# exit()

df = pd.read_csv('weather.csv')

df.columns = ['datetime', 'temperature']
df = df.iloc[::-1]
# print(df.head())

df[['date', 'time']] = df['datetime'].str.split(expand=True)
df[['day', 'month', 'year']] = df['date'].str.split('.', expand=True)
df[['time_h']] = df['time'].str.split(':', expand=True)[0]

# a=df['month'].astype(int).values.tolist()
# print(a)
# print(type(a))
# exit()

# df[['month_s']] = [calendar.month_name[df.loc[el]['month']] for el in df]
month_d = dict(enumerate(calendar.month_abbr))
df[['month_s']] = df['month'].astype(int).map(month_d) + ' ' + df['year']
df[['day_time_s']] = df['day'] + ' ' + df['time_h'] + ':00'
df = df.dropna()

# for column in df.columns:
#     print(column)
#     print(df[column].max())
#     print(df[column].min())
#     if df[column].isnull().values.any():
#         print(df[df[column].isnull()])

# exit()

# df.reset_index()
# print(df.head())
# h = df  # .head(10)
# print(df.tail())
# print(df[['date','time']])
# reg = linear_model.LinearRegression()
# reg.fit(df[['day', 'time_h']], df['temperature'])
# reg.fit(df[['day']], df['temperature'])
# reg.fit(h[['day', 'time_h']], h['temperature'])
# pred = reg.predict([[2, 13]])
# print(reg.coef_)
# pr = reg.predict(h[['day', 'time_h']])
# print(pr)
# print(type(pr))
# h['bestfit'] = pr
# print(h)
# print(type(h))

fig = px.scatter(x=df['datetime'], y=df['temperature'])

# fig = px.scatter(x=h['datetime'], y=h['temperature'])
# fig.add_trace(go.Scatter(x=h['datetime'], y=h['bestfit'], mode='lines'))
fig.show()

# df = df[df['year'] == '2017']
# df = df[df['month'] == '01']
# df = df.loc[df['time_h'] == '15']
# df = df[['day', 'month', 'temperature']]
# df = df[['day', 'time_h', 'temperature']]
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(df.head())
print(df.tail())
print(df.shape)
# exit()
# fig = go.Figure(data=[go.Surface(x=df['day'], z=df['month'], y=df[['temperature']])])
# fig = go.Figure(data=[go.Scatter3d(x=df['day'], y=df['month'], z=df['temperature'], mode='markers+lines')])
# fig = go.Figure(data=[go.Mesh3d(x=df['day'], y=df['month'], z=df['temperature'])])

# fig = go.Figure(data=[go.Mesh3d(x=df['day'], y=df['month'], z=df['temperature'], alphahull=2)])

# fig = go.Figure(data=[go.Surface(x=df['day'], z=df['month'], y=df['temperature'])])
# for y in range(2010, 2021):
#     year = str(y)
#     df1 = df[df['year'] == year]
#     # layout = go.Layout(xaxis_title=year,)
#     layout = go.Layout(title=year)
#     # fig = go.Figure(data=[go.Heatmap(x=df1['day'], y=df1['month'], z=df1['temperature'], zmin=-30, zmax=30)], layout=layout)
#     fig = go.Figure(data=[go.Heatmap(x=df1['day_time_s'], y=df1['month_s'], z=df1['temperature'], zmin=-30, zmax=30)], layout=layout)
#     # fig.update_traces(yaxis=)
#     fig.show()


# df1 = df[(df['year'] == '2017') | (df['year'] == '2018')]
df1 = df[(df['year'] >= '2016')]
# df1 = df
# layout = go.Layout(title=year)
fig = go.Figure(data=[go.Heatmap(x=df1['day_time_s'], y=df1['month_s'], z=df1['temperature'], zmin=-30, zmax=30)])
fig.show()
