import importlib
import numpy as np
import matplotlib.pyplot as plt
from initializer import initialize_framework
from data_preprocessing.data_loader import load_demonstrations
from datasets.dataset_keys import dataset_keys_dic
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.io as pio
import plotly.graph_objects as go
import pickle


def get_number_of_primitives(dataset_name):
    return len(dataset_keys_dic[dataset_name])


def show_dataset(dataset_name, subplots, figsize):
    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    axs = np.array(axs).reshape(-1)
    number_of_primitives = get_number_of_primitives(dataset_name)
    for i in range(number_of_primitives):
        demonstrations = load_demonstrations(dataset_name, str(i))['demonstrations raw']
        for j in range(len(demonstrations)):
            axs[i].scatter(demonstrations[j][0], demonstrations[j][1], color='red')


def evaluate_system(quanti_eval, quali_eval, model_name, dataset_name, demo_id, density, simulated_trajectory_length, results_base_directory, saturate=True):
    results_directory = 'results/final/%s/%s/' % (dataset_name, model_name)
    results_directory = results_base_directory + results_directory
    save_path = 'results_analysis/%s_%s_%s.pdf' % (dataset_name, model_name, str(demo_id))

    # Get parameters
    Params = getattr(importlib.import_module('params.' + model_name), 'Params')
    params = Params(results_base_directory)

    # Modify some parameters
    params.dataset_name = dataset_name
    params.load_model = True
    params.save_evaluation = False
    params.results_path = results_directory + str(demo_id) + '/'
    params.selected_primitives_ids = str(demo_id)
    params.saturate_out_of_boundaries_transitions = saturate
    params.show_plot = True
    params.density = density
    params.simulated_trajectory_length = simulated_trajectory_length
    params.quanti_eval = quanti_eval
    params.quali_eval = quali_eval

    # Initialize framework
    learner, evaluator, data = initialize_framework(params, model_name, verbose=False)

    metrics_acc, metrics_stab = evaluator.run(iteration=0, save_path=save_path)

    return metrics_acc, metrics_stab


def evaluate_system_comparison(quanti_eval, quali_eval, models_names, dataset_name, demo_id, density,
                               simulated_trajectory_length, results_base_directory):
    for i in range(len(models_names)):
        model_name = models_names[i]
        if model_name == 'behavioral_cloning':
            saturate = False
        else:
            saturate = True
        evaluate_system(quanti_eval, quali_eval, model_name, dataset_name, demo_id, density,
                        simulated_trajectory_length, results_base_directory, saturate=saturate)


def get_camera_parameters(dataset_name, model_name, demo_id, model):
    plot_data = pickle.load(
        open('results/final/%s/%s/%i/images/primitive_0_iter_%i.pickle' % (dataset_name, model_name, demo_id, model),
             'rb'))

    fig = go.Figure(data=plot_data['3D_plot'])

    app = dash.Dash()
    app.layout = html.Div([
        html.Div(id="output"),  # use to print current relayout values
        dcc.Graph(id="fig", figure=fig)
    ])

    @app.callback(Output("output", "children"), Input("fig", "relayoutData"))
    def show_data(data):
        # show camera settings like eye upon change
        return [str(data)]

    app.run_server(debug=True, use_reloader=False)


def plot_LASA_S2(dataset_name, model_name, demo_id, model, camera):
    save_path = 'results_analysis/%s_%s_%i.pdf' % (dataset_name, model_name, demo_id)
    plot_data = pickle.load(
        open('results/final/%s/%s/%i/images/primitive_0_iter_%i.pickle' % (dataset_name, model_name, demo_id, model),
             'rb'))
    fig = go.Figure(data=plot_data['3D_plot'])
    fig.update_layout(scene=dict(camera=camera))

    # Save image
    pio.write_image(fig, save_path + '.pdf', width=500, height=500)

    # Show
    fig.show()
