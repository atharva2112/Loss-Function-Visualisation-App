#%%
# Author: Atharva Haldankar
# Date: 11th August,2023
# Description: Visualising loss functions using Dash for Dr.Reza Jafari
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
#%%
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label('A:'),
        dcc.Input(id='a11', type='number', value=1),
        dcc.Input(id='a12', type='number', value=0),
        dcc.Input(id='a21', type='number', value=0),
        dcc.Input(id='a22', type='number', value=1),
        html.Label('b:'),
        dcc.Input(id='b1', type='number', value=0),
        dcc.Input(id='b2', type='number', value=0),
        html.Label('c:'),
        dcc.Input(id='c', type='number', value=0),
    ], style={'display': 'inline-block'}),
    html.Div([
        html.Label('Learning Rate:'),
        dcc.Slider(
            id='learning-rate',
            min=0,
            max=1,
            step=0.01,
            value=0.1,
            marks={i / 10: str(i / 10) for i in range(0, 11)}
        ),
    ], style={'display': 'inline-block', 'width': '400px'}),
    html.Div([
        html.Label('Algorithm:'),
        dcc.Dropdown(
            id='algorithm',
            options=[
                {'label': 'Gradient Descent', 'value': 'gd'},
                {'label': 'Stochastic Gradient Descent', 'value': 'sgd'},
                {'label': 'Gradient Descent with Line Search', 'value': 'gdls'},
                {'label': 'Newton\'s Method', 'value': 'newton'},
            ],
            value='gd',
            style={'width': 300}
        ),
    ], style={'display': 'inline-block'}),
    dcc.Graph(
        id='graph',
        clickData={'points': [{'x': 0, 'y': 0}]},
        style={'width': 800, 'height': 600}
    ),
])


@app.callback(
    Output('graph', 'figure'),
    Input('a11', 'value'),
    Input('a12', 'value'),
    Input('a21', 'value'),
    Input('a22', 'value'),
    Input('b1', 'value'),
    Input('b2', 'value'),
    Input('c', 'value'),
    Input('learning-rate', 'value'),
    Input('algorithm', 'value'),
    Input('graph', 'clickData')
)
def update_graph(a11, a12, a21, a22, b1, b2, c, learning_rate, algorithm, clickData):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(x, y)

    A = np.array([[a11, a12], [a21, a22]])

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xy = np.array([X[i, j], Y[i, j]])
            Z[i, j] = xy.T @ A @ xy + b1 * X[i, j] + b2 * Y[i, j] + c

    trace1 = go.Contour(x=x, y=y, z=Z)

    x0 = clickData['points'][0]['x']
    y0 = clickData['points'][0]['y']

    x_vals = [x0]
    y_vals = [y0]

    for i in range(10):
        if algorithm == 'gd':
            x0 = x0 - learning_rate * (2 * a11 * x0 + a12 * y0 + b1)
            y0 = y0 - learning_rate * (2 * a22 * y0 + a21 * x0 + b2)
        elif algorithm == 'sgd':
            idx = i % 2
            if idx == 0:
                x0 = x0 - learning_rate * (2 * a11 * x0 + a12 * y0 + b1)
            else:
                y0 = y0 - learning_rate * (2 * a22 * y0 + a21 * x0 + b2)
        x_vals.append(x0)
        y_vals.append(y0)
    trace2 = go.Scatter(x=x_vals, y=y_vals, mode='markers+lines', name='Gradient Descent')
    global_min_x = 0
    global_min_y = 0
    trace3 = go.Scatter(
        x=[global_min_x],
        y=[global_min_y],
        mode='markers',
        marker={'size': 10, 'color': 'red'},
        name='Global Minimum'
    )
    local_min_x = x_vals[-1]
    local_min_y = y_vals[-1]
    trace4 = go.Scatter(
        x=[local_min_x],
        y=[local_min_y],
        mode='markers',
        marker={'size': 10, 'color': 'orange'},
        name='Local Minimum'
    )
    return {
        'data': [trace1, trace2, trace3, trace4],
        'layout': go.Layout(
            title=f'f(x,y)={a11}x^2+{a12}xy+{a21}yx+{a22}y^2+{b1}x+{b2}y+{c}',
            xaxis={'title': 'x'},
            yaxis={'title': 'y'}
        )
    }
#%%
# Run the app
app.run_server(
    port=8021,
    host='0.0.0.0'
)