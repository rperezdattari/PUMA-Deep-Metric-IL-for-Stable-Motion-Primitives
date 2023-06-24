import optuna
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.trial import TrialState
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import os
#os.chdir('../results/final/optuna/')

study_name = 'optuna_study_2nd_order_2D_euc_no_bound_LAIR'
study = optuna.load_study(study_name=study_name, storage='sqlite:///%s.db' % study_name)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print('Study statistics: ')
print('  Number of finished trials: ', len(study.trials))
print('  Number of pruned trials: ', len(pruned_trials))
print('  Number of complete trials: ', len(complete_trials))

# Get top 10 trials and plot
trials_scores = []
for i in range(len(complete_trials)):
    trial = complete_trials[i]
    summary = [i, trial.value]
    trials_scores.append(summary)

sorted_scores = np.array(trials_scores)[np.array(trials_scores)[:, 1].argsort()]
top_scores = sorted_scores[:10]

for i in range(top_scores.shape[0]):
    trial_id = int(top_scores[i][0])
    ranking = int(top_scores[i][1])
    trial = complete_trials[trial_id]
    intermediate_values = np.fromiter(trial.intermediate_values.values(), dtype=float)
    smoothed = gaussian_filter1d(intermediate_values, sigma=1)
    plt.plot(smoothed, label='trial %i, ranking %i' % (trial_id, i+1), )

plt.legend()
plt.show()

# Do the rest of optuna stuff
#best_trial = study.best_trial
best_trial = complete_trials[76]
print('Best trial:', best_trial.number)


print('  Value: ', best_trial.value)

print('  Params: ')
for key, value in best_trial.params.items():
    print('    {}: {}'.format(key, value))

#plot_optimization_history(study).show(renderer='browser')
#plot_intermediate_values(study).show(renderer='browser')
#plot_param_importances(study).show(renderer='browser')
