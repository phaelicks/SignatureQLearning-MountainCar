from collections import deque
import copy
from pprint import pprint
import sys
import tqdm.notebook as tqdm
from typing import Optional, Dict, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils

"""
TODO: check if signature removal work as intended 
See reference: Lyons, Differential equations driven by rough paths (2004)
"""

def train(
        env, 
        qfunction, 
        episodes: int = 100, 
        discount: float = 0.99, 
        learning_rate: float = 0.1, 
        learning_rate_decay: Dict[str, Any] = {'mode': None},
        epsilon: float = 0.2,
        epsilon_decay: Dict[str, Any] = {'mode': None},
        decay_increment: str = 'episode',
        batch_size: int = 1,
        window: Optional[int] = None
    ) -> List:
    
    max_steps = env.spec.max_episode_steps  # episode max steps
    successes = 0

    reward_history = []
    loss_history = []
    start_position_history = []
    end_position_history = []
    steps_history = []
    first_obs_value = []
    first_Q_values = []
    intermediate_policies = []

    # EPSILON GREEDY DETAILS
    initial_epsilon = epsilon
    epsilon_decay_lambda = utils.create_decay_schedule(start_value=initial_epsilon, **epsilon_decay)

    # LEARNING DETAILS
    loss_fn = nn.SmoothL1Loss() # nn.MSELoss()
    lr_decay_lambda = utils.create_decay_schedule(start_value=learning_rate, **learning_rate_decay)
    optimizer = optim.Adam(qfunction.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    batch_count = 0
    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        pbar.set_description("{} successes  |  Episodes".format(successes))

        episode_loss = 0
        episode_reward = 0

        # create first observation tuple
        state = env.reset()[0] # still includes velocity as 2nd component
        start_position = state[0]
        start_position_history.append(start_position)
        state = np.hstack((start_position, 1.)) # 1 - 1/(max_steps+1))) # 0.))
        
        # add to history
        if window != None:
            assert window > 0, "History window length must be a positive integer."
            history = deque(maxlen=window)
            history.append(state)            

        # wrap into batch of length 1 with shape (1, 1, channels)
        initial_tuple = torch.tensor(
            [state], requires_grad=False, dtype=torch.float).unsqueeze(0)
        # compute signature of first tuple
        ##########history_signature = qfunction.update_signature(initial_tuple)
        history_signature = qfunction.compute_signature(initial_tuple, with_basepoint=True)
        last_tuple = initial_tuple

        # save first Q-value
        first_Q_values.append(qfunction(history_signature)[0].detach())

        # save value for first observation [-0.5, 0.]
        first_obs = torch.tensor(
            [[-0.5, 1.]], requires_grad=False, dtype=torch.float).unsqueeze(0)
        first_obs_signature = qfunction.compute_signature(first_obs, with_basepoint=True)
        first_obs_value.append(qfunction(first_obs_signature)[0].detach().mean().item())

        # run episode
        for step in range(max_steps):
            # Choose action according to epsilon-greedy policy
            Q = qfunction(history_signature)[0] # unwrap from batch dimension
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, 3)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            
            # take action, create reward signal and new observation tuple
            state_1, reward, terminated, truncated, _ = env.step(action)
            position_1 = state_1[0]
            state_1 = np.array([position_1, 1 - (step + 1) / max_steps])
            reward = (position_1 - 0.5) / max_steps if position_1 < 0.5 \
                else 0.05 * (1 - (step + 1) / max_steps) ** 2            
 

            # wrap into tensor of shape (1, 1, channels)
            new_tuple = torch.tensor([[state_1]], requires_grad=False, dtype=torch.float)
            if window == None:
                history_signature = qfunction.extend_signature(new_tuple, 
                                                            last_tuple.squeeze(0), # shape (1, channels)
                                                            history_signature)
            else:
                history.append(state_1) # keeps history of constant length
                history_signature = qfunction.compute_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0),
                    with_basepoint=False
                )
            # TODO: find way to shorten signature via Chen which takes into account inital basepoint
            
            # Create target Q value for training the qfunction
            Q_target = torch.tensor(reward, dtype=torch.float)            
            if not truncated and not terminated:
                Q1 = qfunction(history_signature)[0]  # unwrap from batch dimension
                maxQ1, _ = torch.max(Q1, -1)
                Q_target += torch.mul(maxQ1, discount)
            Q_target.detach_()

            
            loss = loss_fn(Q[action], Q_target)
            episode_loss += loss.item()
            loss = loss / batch_size
            loss.backward()
            batch_count += 1
            episode_reward += reward

            # Calculate loss and update qfunction            
            if batch_count == batch_size or (terminated or truncated):
                optimizer.step()
                qfunction.zero_grad()
                batch_count=0

            if (terminated or truncated):
                if position_1 >= 0.5:
                    # On successful episodes, adjust the following parameters
                    successes += 1
                    if decay_increment == 'success':
                        scheduler.step()
                        epsilon = initial_epsilon * epsilon_decay_lambda(successes)
                
                #if episode == 0 or episode % 100 == 0:
                #    print(" Final Q-values:", Q.detach())
                    
                # Record history
                loss_history.append(episode_loss)
                reward_history.append(episode_reward)
                end_position_history.append(position_1)
                steps_history.append(step+1) # starts at 0

                # close training environment 
                env.close()
                break
            
            else:
                last_tuple = new_tuple 
        
        if decay_increment == 'episode':
            scheduler.step()
            epsilon = initial_epsilon * epsilon_decay_lambda(episode)

        # reset learning rate
        #if episode == episodes // 2:
        #    optimizer = optim.Adam(qfunction.parameters(), lr=learning_rate)
        #    lr_decay_lambda = utils.create_decay_schedule(start_value=learning_rate, **learning_rate_decay)
        #    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

        if (episode+1) % 10 == 0:
            qfunction_copy = copy.deepcopy(qfunction.state_dict())
            intermediate_policies.append(qfunction_copy)
                
    print('successful episodes: {:d} - {:.4f}%'.format(successes, successes / episodes * 100))    

    return [
        reward_history, 
        loss_history, 
        end_position_history, 
        steps_history, 
        start_position_history, 
        first_obs_value,
        torch.stack(first_Q_values, dim=0),
        intermediate_policies,
    ]

    
def test_multiple_episodes(env, qfunction, episodes, epsilon=0.0):
    reward_history = []
    success_history = []
    steps_history = []
    start_history = []
    first_obs_value_history = []
    observation_histories = []

    #pbar = tqdm.trange(episodes, file=sys.stdout)
    #for episode in pbar:
    #    pbar.set_description("Episode {}".format(episode))
    
    for _ in range(episodes):
        reward, success, steps, start, first_obs_value, observations = test_single_episode(
            env, qfunction, epsilon=epsilon
        )
        reward_history.append(reward)
        success_history.append(success)
        steps_history.append(steps)
        start_history.append(start)
        observation_histories.append(observations)
        first_obs_value_history.append(first_obs_value)
    
    return [
        reward_history,
        success_history,
        steps_history,
        start_history,
        first_obs_value_history,
        observation_histories
    ]

def test_single_episode(env, qfunction, epsilon=0.0, render=False):
    history = []
    max_steps = env.spec.max_episode_steps
    num_steps = 0
    success = False
    episode_reward = 0

    # create first state tuple
    state = env.reset()[0]
    start_position = state[0]
    state = np.array([start_position, 1.]) # record timestep
    history.append(state)

    # signature of first tuple
    initial_tuple = torch.tensor([[state]], requires_grad=False, dtype=torch.float)
    history_signature = qfunction.compute_signature(initial_tuple, with_basepoint=True)
    last_tuple = initial_tuple

    # value of first observation (-0.5, 0) (or (-5.0, 1.))
    first_obs = torch.tensor(
        [[-0.5, 1.]], requires_grad=False, dtype=torch.float).unsqueeze(0)
    first_obs_signature = qfunction.compute_signature(first_obs, with_basepoint=True)
    first_obs_value = qfunction(first_obs_signature)[0].detach().mean().item()

    
    # run episode
    for step in range(max_steps):
        if render:
            env.render()

        # select action
        Q = qfunction(history_signature)[0] # unwrap from batch
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            _, action = torch.max(Q, -1)
            action = action.item()        
        
        # take action
        state, _, terminated, truncated, _ = env.step(action)
        position = state[0]
        episode_reward += (position - 0.5) / max_steps if position < 0.5 \
            else 0.05 * (1 - (step + 1) / max_steps) ** 2

        # create new observation tuple
        state = np.array([position, 1 - (step + 1)/ max_steps]) # Record the timestep
        history.append(state)
        new_tuple = torch.tensor([[state]], requires_grad=False, dtype=torch.float)

        # update signature
        history_signature = qfunction.extend_signature(new_tuple, 
                                                       last_tuple.squeeze(0),
                                                       history_signature)

        if (terminated or truncated):
            success = (position >= 0.5)
            num_steps = step + 1 # starts at 0
            env.close()
            break
        else:
            last_tuple = new_tuple

            
    return episode_reward, success, num_steps, start_position, first_obs_value, np.array(history)