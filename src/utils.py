from random import seed
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import manual_seed
from torch.backends import cudnn

from sigqlearning_policies import SigQFunction

#--------------------------------------------------
# plot functionalities
#--------------------------------------------------

def plot_results(results_all_runs, run=-1, ma_window=100, title=None, show=True, figsize=None):
    results = results_all_runs[run]

    y_labels = ['reward', 'loss', 'end position', 'steps to termination']
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained', figsize=figsize)
    for ax, id in zip(axes.flat, range(4)):
        res = ax.plot(results[id], label=y_labels[id])
        ma = ax.plot(pd.Series(results[id]).rolling(ma_window).mean(), label="SMA {}".format(ma_window))
        ax.set_ylabel(y_labels[id])
        if id > 1:
            ax.set_xlabel('episodes')
        if id==1:
            fig.legend(handles=ma, loc='outside upper right')  
            
    fig.suptitle(title)
    
    if show:
        plt.show()
    else: 
        return fig

def plot_mean_results(means, stds, ma_window=0, title=None, show=True, figsize=(6,4.5)):
    y_labels = ['reward', 'loss', 'end position', 'steps to termination']
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained', figsize=figsize)
    for ax, id in zip(axes.flat, range(4)):
        ax.plot(means[id], color="b", label="mean over runs") 
        ax.plot(pd.Series(means[id]).rolling(ma_window).mean())
        ax.fill_between(range(len(means[id])),
                        means[id] - 1 * stds[id],
                        means[id] + 1 * stds[id],
                        color='b', alpha=0.2, label='+/- one standard deviation')
        ax.set_ylabel('mean ' + y_labels[id])
        if id > 1:
            ax.set_xlabel('episodes')
        #ax.legend(loc="best") # legend in each subplot
        if id==0:
            fig.legend(bbox_to_anchor=(0.18, 1.02, 1., .102), loc='lower left',
                      ncols=2, borderaxespad=-0.2)
    fig.suptitle(title, y=1.127)

    if show:
        plt.show()
    else: 
        return axes

def save_mean_results_plots(means, stds, file_path, file_id, title=False,
                            ma_window=0, figsize=(5.5,4.125), show=False):
    y_labels = ['reward', 'loss', 'end position', 'steps to termination']
    for y_label, id in zip(y_labels, range(4)):
        plt.figure(figsize=figsize)        
        plt.plot(means[id], color="b", label='mean {}'.format(y_label))
        plt.plot(pd.Series(means[id]).rolling(ma_window).mean())   
        if y_label == 'end position':
            fill_lower = means[id] - stds[id]
            fill_upper = [mean + std if mean + std <= 0.5 else mean for mean, std in zip(means[id], stds[id])]
        elif y_label == 'steps to termination':
            fill_lower = means[id] - stds[id]
            fill_upper = [min(200, mean + std) for mean, std in zip(means[id], stds[id])]
        else:   
            fill_lower = means[id] - stds[id]
            fill_upper = means[id] + stds[id]
        plt.fill_between(range(len(means[id])), fill_lower, fill_upper,
                         color='b', alpha=0.2, label='+/- one standard deviation')
        plt.ylabel(y_label, fontsize=11)
        plt.xlabel('episode', fontsize=11)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)        
        plt.legend(loc='lower left' if y_label == 'steps to termination' else 'lower right',
                   fontsize=11)
        if title:
            plt.title('Average {} over training runs'.format(y_label), fontsize=10)
        else: 
            plt.title(' ', fontsize=12)
        plt.tight_layout()
        plt.savefig(file_path + file_id + '_mean_' + y_label + '.png')
        if show:
            plt.show()
        else:
            plt.close()


def plot_specific_result(result_to_print, results, ma_window=100, point=False):
    names = [
        "reward", "loss", "end position", "steps", "start position", "first Q-value"
    ]
    assert result_to_print in names, "Select existing result to plot."
    # TODO: change to key-value approach 
    
    result_id = names.index(result_to_print)
    result_list = results[result_id]
    plt.plot(result_list)
    plt.plot(pd.Series(result_list).rolling(ma_window).mean())
    plt.ylabel(result_to_print)
    plt.xlabel("episode")
    #plt.tight_layout()
    plt.show()


def scatter_plot_positions(results):
    plt.scatter(results[4], results[2])
    plt.ylabel("End position (>= 0.5 is a success)")
    plt.xlabel("Start position")
    plt.title("Start vs. end positions")
    plt.show()  


def plot_first_Q_values(results_all_runs, run_id=-1, step=10, time='down', window=None,
                        intermediate_qfunctions=False, sigq_container=None,
                        start_position = None, show=True, 
                        save=False, file_path=None):
    if save: 
        assert file_path != None, \
            "Provide file_path to save plot under."
    if intermediate_qfunctions:
        assert (sigq_container != None), \
            "Provide instance of SigQFunction as sigq_container"
        assert (len(results_all_runs[run_id][-1]) > 0), \
            "No saved intermediate Q-function in results."
    
    t_0 = 0. if time=='up' else 1.
    
    run_results = results_all_runs[run_id]
    start, end = (0, len(run_results[0])-1) if window==None else (window[0], window[1])

    if intermediate_qfunctions:
        # with intermediate qfunctions
        if start_position == None:
            initial_tuples = [
                torch.tensor([[position_0, t_0]], requires_grad=False, dtype=torch.float).unsqueeze(0)
                for position_0 in run_results[4][9::step]
            ]
        else:
            initial_tuples = [
                torch.tensor([[start_position, t_0]], requires_grad=False, dtype=torch.float).unsqueeze(0)
                for _ in range(len(run_results[-1]))
            ]

        first_Q_values = []
        for intermediate_qfunction, initial_tuple in zip(run_results[-1], initial_tuples):
            sigq_container.load_state_dict(intermediate_qfunction)
            sigq_container.eval()
            first_signature = sigq_container.compute_signature(initial_tuple, 
                                                               with_basepoint=True)
            first_Q_values.append(sigq_container(first_signature)[0].detach())
        first_Q_values = torch.stack(first_Q_values, dim=0)
        xaxis = [i*10 for i in range(len(first_Q_values))]
        #[i*step for i in range(len(first_Q_values[start//10: end//10]))]
        plt.plot(xaxis[start//10:end//10], first_Q_values[start//10:end//10])
        plt.plot(xaxis[start//10:end//10], first_Q_values[start//10:end//10].mean(dim=-1), 
                 color='black', alpha=0.8, linestyle='dashed')  

    else:
        # with saved Q values 
        first_Q_values = run_results[-2]
        first_Q_values = torch.stack(
            [q_values for q_values in first_Q_values[start:end:step]], dim=0
        )
        xaxis = [i*step for i in range(len(first_Q_values))]
        plt.plot(xaxis, first_Q_values)
        plt.plot(xaxis,first_Q_values.mean(dim=-1), 
                 color='black', alpha=0.8, linestyle='dashed')

    plt.xlabel("episode")
    first_tuple_str = " for ({}, {})".format(start_position, t_0) if start_position != None else ""
    plt.title("First Q-values during training run {}".format(run_id) + first_tuple_str)
              #+ " for ({},0)".format(start_position) if start_position != None else "")
    plt.legend(
        ["Action " + str(i) for i in range(first_Q_values.shape[-1])] 
        + ["Mean Q-value"], loc="best"
    )     

    if save:
        plt.savefig(file_path)
        print('Plot saved under \"{}\".'.format(file_path))
    if show:
        plt.show()          
    plt.close()


def plot_first_obs_value(means, stds, ma_window=0, show=True, 
                         save=False, file_path=None, figsize=(6,4.5)):
    
    plt.plot(means, color="b", label='mean value')
    plt.plot(pd.Series(means).rolling(ma_window).mean())        
    plt.fill_between(range(len(means)), means - 1 * stds, means + 1 * stds,
                     color='b', alpha=0.2, label='+/- one standard deviation')
    plt.xlabel('episodes')
    plt.ylabel('value')
    plt.title('Average value of first observation (-0.5, 0) during training', fontsize=11)
    plt.legend(loc='best')

    if save:
        plt.savefig(file_path)
        print('Plot saved under \"{}\".'.format(file_path))
    if show:
        plt.show()
    plt.close()

#--------------------------------------------------
# hyperparameter decay schedules
#--------------------------------------------------

def create_decay_schedule(mode=None, start_value=None, **kwargs):
    if mode == None:
        return lambda step: 1
    elif mode == 'linear':
        return linear_decay(start_value=start_value, **kwargs)
    elif mode == 'exponential':
        return exponential_decay(**kwargs)
    else:
        raise ValueError('Provide mode as None, linear or exponential.')

def exponential_decay(factor=1, end_value=0., wait=0):
    assert 0 < factor <= 1, "multiplicative decay :factor: in :mode: 'exponential' \
                            must be in intervall [0, 1]."
    assert 0 <= wait, ":wait: epochs must be greater or equal to zero."
    
    # return multiplicative decay factor
    return lambda epoch: (
        1 if (
            epoch < wait
        ) else ( 
            max(factor ** epoch, end_value))
    )
     
def linear_decay(epochs, start_value, end_value=0., wait=0, steps=None):
    assert epochs > 0, "number of total :epochs: to decay over in :mode: 'linear' \
                        must be an integer greater than zero."
    assert 0 <= wait <= epochs, ":wait: epochs must be an integer between 0 and :epochs:"

    if steps == None:
        steps = epochs - wait
    else:
        assert steps > 0, "number of :steps: to decay over in :mode: 'linear' \
                            must be an integer greater than zero."
    
    epochs -= wait 
    frac = end_value / start_value    
    step_length = np.ceil(epochs / (steps+1))
    
    # return factor to decay :start: to :end: linearly in :steps: over :epochs: - :wait:
    # assuming :epoch: is counted from 0 to :epochs:-1
    return lambda epoch: (
        1 if (
            epoch < wait
        ) else ( 
            1 - (1 - frac) * min( ((epoch - wait) // step_length) / steps, 1))
        )      

#--------------------------------------------------
# other utilities
#--------------------------------------------------  

def make_reproducable(base_seed=0, numpy_seed=0, torch_seed=0):
    seed(base_seed)
    np.random.seed(numpy_seed)
    manual_seed(torch_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
