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
import pandas as pd
import seaborn as sns


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


def evaluate_system(quanti_eval, quali_eval, model_name, dataset_name, demo_id, density, simulated_trajectory_length,
                    evaluation_samples_length, results_base_directory, fixed_point_iteration_thr=None, saturate=True):
    results_directory = 'results/final/%s/%s/' % (dataset_name, model_name)
    results_directory = results_base_directory + results_directory
    save_path = results_base_directory + 'results_analysis/%s_%s_%s.pdf' % (dataset_name, model_name, str(demo_id))

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
    params.evaluation_samples_length = evaluation_samples_length

    if fixed_point_iteration_thr is not None:
        params.fixed_point_iteration_thr = fixed_point_iteration_thr

    # Initialize framework
    learner, evaluator, data = initialize_framework(params, model_name, verbose=False)

    metrics_acc, metrics_stab = evaluator.run(iteration=0, save_path=save_path)

    return metrics_acc, metrics_stab, evaluator


def evaluate_system_comparison(quanti_eval, quali_eval, models_names, dataset_name, demos_ids, density,
                               simulated_trajectory_length, evaluation_samples_length, results_base_directory,
                               fixed_point_iteration_thr=None, save_name=None):
    metrics_models = {}
    for model_name in models_names:
        RMSE, DTWD, FD = [], [], []
        n_spurious = []
        for demo_id in demos_ids:
            if quanti_eval:
                print('Evaluating model: %s; demo: %i' % (model_name, demo_id))

            if model_name == 'behavioral_cloning':
                saturate = False
            else:
                saturate = True
            metrics_acc, metrics_stab, evaluator = evaluate_system(quanti_eval, quali_eval, model_name, dataset_name,
                                                                   demo_id, density, simulated_trajectory_length,
                                                                   evaluation_samples_length, results_base_directory,
                                                                   fixed_point_iteration_thr=fixed_point_iteration_thr,
                                                                   saturate=saturate)
            if quanti_eval:
                RMSE = RMSE + evaluator.RMSE[-1]
                DTWD = DTWD + evaluator.DTWD[-1]
                FD = FD + evaluator.FD[-1]
                n_spurious.append(metrics_stab['n spurious'])

        metrics_model = {'RMSE': RMSE,
                         'DTWD': DTWD,
                         'FD': FD,
                         'n_spurious': n_spurious}

        metrics_models[model_name] = metrics_model

    if quanti_eval:
        with open(results_base_directory + 'results_analysis/saved_metrics/%s.pk' % save_name, 'wb') as file:
            pickle.dump(metrics_models, file)

    return metrics_models


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


def plot_accuracy_metrics(models_names, metrics_names, metrics_models, title, results_base_directory, colors=None, unit='mm'):
    #plt.rcParams.update({'font.size': 14})

    column_names = metrics_names
    df = pd.DataFrame(columns=column_names)

    i = 0
    for metrics_model in metrics_models:
        n_demonstrations = len(metrics_model['RMSE'])
        for metric in metrics_names:
            metric_data = pd.DataFrame({'Metric': np.repeat(metric, n_demonstrations),
                                        'Error (%s)' % unit: metrics_model[metric],
                                        'Model': np.repeat(models_names[i], n_demonstrations)})
            df = pd.concat([df, metric_data])
        i += 1
    sns.set_theme(style='whitegrid', rc={'grid.linestyle': '--', 'text.usetex': True, "font.family": "Times New Roman"})
    if colors is None:
        palette = sns.color_palette('tab10')
    else:
        palette = colors

    PROPS = {
        'boxprops': {'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }

    metrics_plot = sns.boxplot(x='Metric', y='Error (%s)' % unit, hue='Model', data=df, linewidth=3, showfliers=False,
                               width=0.7, zorder=3, palette=palette, **PROPS)
    metrics_plot.tick_params(labelsize=10)
    plt.title(title, y=1, fontsize=20)
    plt.tight_layout()
    plt.savefig(results_base_directory + 'results_analysis/box_plot_%s.pdf' % title)
    plt.show()


def boundary_evaluation(model_name, dataset_name, demos_ids, density, results_base_directory):
    results_directory = 'results/final/%s/%s/' % (dataset_name, model_name)
    results_directory = results_base_directory + results_directory

    # Get parameters
    Params = getattr(importlib.import_module('params.' + model_name), 'Params')
    params = Params(results_base_directory)

    # Modify some parameters
    params.dataset_name = dataset_name
    params.load_model = True
    params.save_evaluation = False
    params.saturate_out_of_boundaries_transitions = False
    params.show_plot = True
    params.density = density
    losses = []

    for demo_id in demos_ids:
        # Get id
        params.selected_primitives_ids = str(demo_id)
        params.results_path = results_directory + str(demo_id) + '/'

        # Initialize framework
        learner, evaluator, data = initialize_framework(params, model_name, verbose=False)

        # initial states
        learner.batch_size = density ** params.manifold_dimensions
        state_sample, primitive_type_sample_gen = learner.space_sample()

        # get cost
        learner.boundary_loss_weight = 1
        loss = learner.boundary_constrain(state_sample, primitive_type_sample_gen).cpu().detach().numpy()
        losses.append(loss)

    return np.array(losses)