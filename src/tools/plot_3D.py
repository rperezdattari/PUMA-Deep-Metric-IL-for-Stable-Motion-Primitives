import pickle
import plotly.graph_objects as go
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

iteration = sys.argv[1]
plot_data = pickle.load(
    open('results/final/LASA_S2/1st_order_S2/22/images/primitive_0_iter_%s.pickle' % iteration, 'rb'))

fig = go.Figure(data=plot_data['3D_plot'])

camera = dict(
    eye=dict(x=-0.8550474526258948, y=-0.8632259816571023, z=0.8694060571650828),
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)
)
fig.update_layout(scene=dict(camera=camera))

fig.show()

app = dash.Dash()
app.layout = html.Div([
    html.Div(id="output"),  # use to print current relayout values
    dcc.Graph(id="fig", figure=fig)
])


@app.callback(Output("output", "children"), Input("fig", "relayoutData"))
def show_data(data):
    # show camera settings like eye upon change
    return [str(data)]


app.run_server(debug=False, use_reloader=False)
