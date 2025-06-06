import numpy as np
import time

# Define algorithm parameters
alpha = .9
gamma = .9

# Define states and actions
# States are the integers between 0 and 9
# Actions are to increase state by 1 or decrease it by -1
# Periodic conditions are suitable
states = np.arange(0, 10)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
actions = np.array(['Decrease', 'Increase', 'Stay', 'Jump2'])
# Decrease += -1, Increase += 1, Stay += 0 

# Generate matrix of zeros with n_states x n_actions
n_states = len(states)
n_actions = len(actions)
Q_matrix = np.zeros([n_states, n_actions])
print(Q_matrix)

# Define training parameters
N_steps = 100000
epsilon = 0.05

def environment(state, action):
    if action == 'Increase':
        new_state = (state + 1) % (n_states)
    elif action == 'Decrease': 
        new_state = (state - 1) * (state > 0)
    elif action == 'Stay':
        new_state = state
    elif action == 'Jump2':
        new_state = (state + 2) % n_states
    return new_state

def get_update_q_function(Q_old, reward):
    # policy update / improvement
    return Q_old + alpha * (reward + np.max(Q_matrix[state]) * gamma - Q_old)

def get_action(Q_matrix, state):
    # policy evaluation
    maxQ = np.max(Q_matrix[state])
    # get action index associated with Q value
    argmax = np.random.choice(np.where(Q_matrix[state] == maxQ)[0])
    return argmax

# Initialize state
state = 0
for i in range(N_steps):
    if np.random.uniform(0, 1) < epsilon:
        # If falls into epsilon, action is random
        action_index = np.random.choice(np.arange(len(actions)))
    else:
        action_index = get_action(Q_matrix, state)
    
    # update state according to environment
    new_state = environment(state, actions[action_index])

    # reward is simply the new state
    reward = new_state

    # get index of action and state for update
    # state index is just the own state
    state_index = state

    # update q table
    Q_matrix[state_index][action_index] = get_update_q_function(Q_matrix[new_state][action_index], reward) 
    
    # update state
    state = new_state

print(Q_matrix)

time.sleep(1)

# Evaluating the trained policy
state = 0
for i in range(25):
    action_index = get_action(Q_matrix, state)
    # update state according to environment
    new_state = environment(state, actions[action_index])
    # update state
    state = new_state
        
    print(state)
    print(states)
    print('\n')
