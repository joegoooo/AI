import numpy as np

class BanditEnv:
    def __init__(self, num_arms, stationary):
        self.num_arms = num_arms
        self.action_history = []
        self.reward_history = []
        self.arms = []
        self.mean_values = []
        self.stationary = stationary  
        self.optimal = 0

    def reset(self):
        self.action_history.clear()
        self.reward_history.clear()
        self.arms.clear()
        self.mean_values = np.random.standard_normal(self.num_arms)
        for i in range(self.num_arms):
            values = np.random.normal(self.mean_values[i], 1, 100)
            self.arms.append(values)
        self.optimal = np.argmax(self.mean_values)

    def step(self, action):
        self.action_history.append(action)
        reward = np.random.choice(self.arms[action])
        self.reward_history.append(reward)
        if not self.stationary:
            # make mean values shift
            random_walks = np.random.normal(0, 0.01, self.num_arms)
            for i in range(self.num_arms):
                self.mean_values[i] += random_walks[i]
            self.arms.clear()
            for i in range(self.num_arms):
                values = np.random.normal(self.mean_values[i], 1, 100)
                self.arms.append(values)
        return reward
    
    def export_history(self):
        return self.action_history, self.reward_history


    