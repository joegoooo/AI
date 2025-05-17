from BanditEnv import BanditEnv
from Agent import Agent
import matplotlib.pyplot as plt

num_arms = 10
num_runs = 2000
num_steps = 1000 # Part 3: 1000, Part 5/7: 10000

alpha = None # Part 3/5: None, Part 7: 0.1
epsilons = [0, 0.01, 0.1] # 0.3 and 0.5 for discussion

# Part 3: stationary=True, Part 5/7: stationary=False
env = BanditEnv(num_arms=num_arms)
env.reset()
agents = [
    Agent(num_arms=num_arms, epsilon=epsilons[i], alpha=alpha) for i in range(len(epsilons))
]

average_rewards = [[0 for i in range(num_steps)] for j in range(len(agents))]
percent_of_optimals = [[0 for i in range(num_steps)] for j in range((len(agents)))]


for j in range(num_runs):
    
    # epsilon = 0
    for k in range(len(agents)):
        env.reset()
        agents[k].reset()
        for i in range(num_steps):
            optimal = env.optimal
            action = agents[k].select_action()
            reward = env.step(action)
            agents[k].update_q(action, reward)
            if action == optimal:
                percent_of_optimals[k][i] += 1
        
        action_history, reward_history = env.export_history()
        for i in range(num_steps):
            average_rewards[k][i] += reward_history[i]

    


for i in range(num_steps):
    for j in range(len(agents)):
        average_rewards[j][i] /= num_runs
        percent_of_optimals[j][i] /= num_runs / 100


plt.figure(1)
plt.plot(average_rewards[0], 'g')
plt.plot(average_rewards[1], 'b')
plt.plot(average_rewards[2], 'r')
# for discussion
# plt.plot(average_rewards[3], 'c')
# plt.plot(average_rewards[4], 'm')
plt.legend([f'epsilon={epsilons[i]}' for i in range(len(epsilons))])
plt.title(f'Average Rewards')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.savefig("AveRewards.png")

plt.figure(2)
plt.plot(percent_of_optimals[0], 'g')
plt.plot(percent_of_optimals[1], 'b')
plt.plot(percent_of_optimals[2], 'r')
# for discussion
# plt.plot(percent_of_optimals[3], 'c')
# plt.plot(percent_of_optimals[4], 'm')
plt.legend([f'epsilon={epsilons[i]}' for i in range(len(epsilons))])
plt.title(f'percentage of Optimal Move')
plt.xlabel('Steps')
plt.ylabel('percentage')
plt.savefig("Optimal.png")

