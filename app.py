from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.graph_objs as go
import json
from scipy.stats import norm, uniform, expon, poisson, binom

app = Flask(__name__)

# Function to generate the plot using Plotly
def plot_distribution(distribution, params):
    x = np.linspace(-10, 10, 1000)
    y = []

    if distribution == "normal":
        mu, sigma = params.get('mu', 0), params.get('sigma', 1)
        y = norm.pdf(x, mu, sigma)
    elif distribution == "uniform":
        a, b = params.get('a', -1), params.get('b', 1)
        y = uniform.pdf(x, a, b - a)
    elif distribution == "exponential":
        scale = params.get('scale', 1)
        x = np.linspace(0, 10, 1000)
        y = expon.pdf(x, scale=scale)
    elif distribution == "poisson":
        lam = params.get('lambda', 3)
        x = np.arange(0, 20)
        y = poisson.pmf(x, lam)
    elif distribution == "binomial":
        n, p = params.get('n', 10), params.get('p', 0.5)
        x = np.arange(0, n+1)
        y = binom.pmf(x, n, p)

    # Create a Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{distribution.capitalize()} Distribution'))
    
    fig.update_layout(
        title=f'{distribution.capitalize()} Distribution',
        xaxis_title='x',
        yaxis_title='Density',
        template='plotly_dark'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    data = request.get_json()
    distribution = data.get('distribution', 'normal')
    params = data.get('params', {})

    plot_json = plot_distribution(distribution, params)
    return jsonify(plot_json)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

