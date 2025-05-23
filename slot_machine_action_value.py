import numpy as np
import matplotlib.pyplot as plt

N_levers = 15

levers = np.array([i for i in range(0, N_levers)])

constant_probs = np.array([i/N_levers for i in levers])

rewards_dict = {}

for lever in levers:
    # primeira entrada da lista é a recompensa, segunda é o numero de vezes que a açao foi tomada (escolher a alavanca)
    rewards_dict[lever] = np.array([0, 0])

Q_approx = np.zeros(15)

while True:
    print("\n\nAlavancas:")
    print(levers)
    print("Funçao de valor:")
    print(Q_approx)

    print("Escolha uma alavanca:")
    lever = int(input())

    if lever == -1:
        break

    if np.random.uniform() < constant_probs[lever]:
        print("Ganhou!")
        reward = 1
    else:
        print("Perdeu no tigrinho!")
        reward = 0

    rewards_dict[lever][0] += reward 
    rewards_dict[lever][1] += 1

    total_rewards = rewards_dict[lever][0]
    total_tries   = rewards_dict[lever][1]

    Q_approx[lever] = total_rewards / total_tries 

plt.plot(levers, Q_approx)
plt.savefig("hist.png", dpi = 400)
