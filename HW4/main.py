from BanditEnv import BanditEnv
from Agent import Agent
import matplotlib.pyplot as plt

env = BanditEnv(10, stationary=False)
env.reset()
alpha = 0.1
agent_0 = Agent(10, 0, alpha)
agent_01 = Agent(10, 0.1, alpha)
agent_001 = Agent(10, 0.01, alpha)
num_runs = 2000
num_steps = 10000 # Part 3: 1000, Part 5: 10000
average_rewards_0 = [0 for i in range(num_steps)]
average_rewards_01 = [0 for i in range(num_steps)]
average_rewards_001 = [0 for i in range(num_steps)]

percent_of_optimal_0 = [0 for i in range(num_steps)]
percent_of_optimal_01 = [0 for i in range(num_steps)]
percent_of_optimal_001 = [0 for i in range(num_steps)]

ave_opt = 0
# Part 3
for j in range(num_runs):
    
    # epsilon = 0
    optimal = env.optimal
    ave_opt += env.mean_values[optimal]
    for i in range(num_steps):
        action = agent_0.select_action()
        reward = env.step(action)
        agent_0.update_q(action, reward)
    action_history, reward_history = env.export_history()
    for i in range(num_steps):
        average_rewards_0[i] += reward_history[i]
        if action_history[i] == optimal:
            percent_of_optimal_0[i] += 1
    env.reset()

    # epsilon = 0.1
    optimal = env.optimal
    for i in range(num_steps):
        action = agent_01.select_action()
        reward = env.step(action)
        agent_01.update_q(action, reward)
    action_history, reward_history = env.export_history()
    for i in range(num_steps):
        average_rewards_01[i] += reward_history[i]
        if action_history[i] == optimal:
            percent_of_optimal_01[i] += 1
    env.reset()

    # epsilon = 0.01
    optimal = env.optimal
    for i in range(num_steps):
        action = agent_001.select_action()
        reward = env.step(action)
        agent_001.update_q(action, reward)
    action_history, reward_history = env.export_history()
    for i in range(num_steps):
        average_rewards_001[i] += reward_history[i]
        if action_history[i] == optimal:
            percent_of_optimal_001[i] += 1
    env.reset()
    agent_0.reset()
    agent_01.reset()
    agent_001.reset()

for i in range(num_steps):
    average_rewards_0[i] /= num_runs
    percent_of_optimal_0[i] /= num_runs / 100
    average_rewards_01[i] /= num_runs
    percent_of_optimal_01[i] /= num_runs / 100
    average_rewards_001[i] /= num_runs
    percent_of_optimal_001[i] /= num_runs / 100

plt.figure(1)
plt.plot(average_rewards_0, 'g')
plt.plot(average_rewards_01, 'r')
plt.plot(average_rewards_001, 'b')
plt.legend(['epsilon=0', 'epsilon=0.1', 'epsilon=0.01'])
plt.title(f'Average Rewards')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.savefig("AveRewards.png")

plt.figure(2)
plt.plot(percent_of_optimal_0, 'g')
plt.plot(percent_of_optimal_01, 'r')
plt.plot(percent_of_optimal_001, 'b')
plt.legend(['epsilon=0', 'epsilon=0.1', 'epsilon=0.01'])
plt.title(f'percentage of Optimal Move')
plt.xlabel('Steps')
plt.ylabel('percentage')
plt.savefig("Optimal.png")

