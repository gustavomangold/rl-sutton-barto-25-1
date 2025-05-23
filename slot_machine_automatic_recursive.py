import numpy as np
import matplotlib.pyplot as plt

N_levers = 15

levers = [i for i in range(0, N_levers)]

constant_probs = [i/N_levers for i in levers]

rewards_dict = {}

for lever in levers:
    # primeira entrada da lista é a recompensa, segunda é o numero de vezes que a açao foi tomada (escolher a alavanca)
    rewards_dict[lever] = [0, 0]

Q_approx = np.zeros(15)
epsilon = .1

n_steps = 50000

for tries in range(0, n_steps):
    
    if np.random.uniform() < .1:
        lever = np.random.randint(0, 14)
    else:
        lever = np.where(levers == np.max(levers))[0][0]

    if np.random.uniform() < constant_probs[lever]:
        reward = 1
    else:
        reward = 0

    rewards_dict[lever][0] += reward 
    rewards_dict[lever][1] += 1

    total_rewards = rewards_dict[lever][0]
    total_tries   = rewards_dict[lever][1]

    Q_approx[lever] = total_rewards / total_tries 

plt.plot(levers, Q_approx)
plt.savefig("epsilon-greedy-automated-hist-steps={}.png".format(n_steps), dpi = 400)
