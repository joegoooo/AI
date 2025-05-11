import numpy as np

class Agent:

    def __init__(self, num_arms, epsilon, alpha=None):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.actions = [0 for i in range(self.num_arms)]
        self.mean_values = [0 for i in range(self.num_arms)]

    def select_action(self):
        p = np.random.random()
        if self.epsilon <= p:
            return np.argmax(self.mean_values)
        else:
            return np.random.randint(0, self.num_arms)


    def update_q(self, action, reward):
        self.actions[action] += 1
        if self.alpha is None:
            self.mean_values[action] = (self.mean_values[action]*(self.actions[action]-1)+reward)/self.actions[action]
        else:
            self.mean_values[action] = self.mean_values[action] + self.alpha * (reward - self.mean_values[action])
        

    def reset(self):
        self.mean_values = [0 for i in range(self.num_arms)]
        self.actions = [0 for i in range(self.num_arms)]