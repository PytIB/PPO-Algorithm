from ppo import Actor, ValueNetwork
import numpy as np

class PPO:
    def __init__(self) -> None:
        
        #self.Maze = [['S',0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.maze = np.array([
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]
                ])
        self.alpha = 0.01 # learning rate
        self.gamma = 0.9 # discount rate 
        self.epsilon = 0.8 #exploration rate

        self.input_size = 16   # 4 x 4 maze

        self.output_size = 4  # 4 possible actions

        self.hidden_size = 64  # number of neurons in a hidden layer
       
        self.goal = (3,3)   # the goal of the agent is to reach (3,3) top right most cell in the maze without landing on 1's
        self.max_step_per_episode = 100


        self.actor = Actor(self.input_size,self.hidden_size,self.output_size)

        self.critic = ValueNetwork(self.input_size,self.hidden_size)


    def run_episode(self,number_of_episodes):

        for episode in range(number_of_episodes):

            observations,actions,rewards,old_probs = [],[],[],[]
            observation = (0,0)
            for step in range(self.max_step_per_episode):

                action_probs = self.actor.forward(observation)
                action = np.random.choice(np.arange(self.output_size), p=action_probs)

                


                # Record the state, action, and action probabilities
                observations.append(observation)
                actions.append(action)
                old_probs.append(action_probs[action])
                
                # Take the action and observe the next state and reward
                if action == 0:  # Up
                    next_observation = (max(observation[0] - 1, 0), observation[1])
                elif action == 1:  # Down
                    next_observation = (min(observation[0] + 1, 3), observation[1])
                elif action == 2:  # Left
                    next_observation = (observation[0], max(observation[1] - 1, 0))
                else:  # Right
                    next_observation = (observation[0], min(observation[1] + 1, 3))
                

                if observation == self.goal:
                    reward = 1  # positive reward if the agent lands on (3,3) and wins the maze 
                    rewards.append(reward)
                    break   # we terminate the episode
                else:
                    reward = -1 if self.maze[next_observation] == 1 else 0      # if the agent lands on 1 cell  reward is -1  otherwise reward is 0 
                    rewards.append(reward)

                # calculates advantages
                advantages = self.calculate_advantages(rewards)
                # Move to the next state
                observation = next_observation
                self.print_results(action,observations,actions,old_probs,rewards,advantages)

    def calculate_advantages(self, rewards):
        # Estimate advantages using the critic network
        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            advantage = advantage * self.gamma + rewards[i]
            advantages.insert(0, advantage)
        return advantages

    def print_results(self,action,observations,actions,old_probabilities,reward,advantages):
        print("Moves:",action)
        
        


myPPO = PPO()


myPPO.run_episode(1)