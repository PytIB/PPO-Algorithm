import numpy as np


class Q_learning:
    def __init__(self) -> None:
        self.STATE = 16
        self.ACTIONS = 4
        self.q_table = np.random.uniform(low=-1,high=1,size=(self.STATE,self.ACTIONS))
        self.all_actions = [(1,0),(0,1),(-1,0),(0,-1)]
        self.ALPHA = 0.6 # learning rate
        self.GAMMA = 0.9 # discount rate 
        self.EPSILON = 0.8 #exploration rate
        self.agent_state = (0,0)
        self.agent = 0
        self.state_counter = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.episode_counter = 0
        self.episode_over = False
        self.Maze = [['S',0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.wrong_move = False
    def update_q(self,state, action, reward, next_state):
        self.q_table[state][action] = self.q_table[state][action] + self.ALPHA * (reward + self.GAMMA * np.max(self.q_table[next_state]) - self.q_table[state][action])
    


    def reward(self,state,action):
        if state[0] + action[0] > 3 or state[0] + action[0] < 0 or state[1] + action[1] > 3 or state[1] + action[1] < 0:
            self.wrong_move = True
            return -10
        
        elif state[0] + action[0] == 3 and state[1] + action[1] == 3:
            self.episode_over = True
            print("Agent WON")
            return 50
            
        elif state[0] + action[0] == 1 and state[1] + action[1] == 2:
            return -10
        else:
            return 0
    

    def generate_legal_actions(self,state):
        legal_actions = []
        for i in range(len(self.all_actions)):
            x,y = self.all_actions[i]
            if not (state[0] + x > 3 or state[0] + x < 0 or  state[1] + y < 0 or  state[1] + y > 3):
                legal_actions.append(i)
        
        return legal_actions

    def generate_move(self):
        if np.random.rand() < self.EPSILON:
            legal_actions = self.generate_legal_actions(self.agent_state)
            print("CURRENT STATE:",self.agent_state)
            print("Legal Actions:",legal_actions)
            raw_action = np.random.randint(0,len(legal_actions))
            print("RAW ACTION:",raw_action)
            action = self.all_actions[legal_actions[raw_action]]
            print("ACTION:",action)
            return action
        
        else:
            print("CURRENT STATE:",self.agent_state)
            agent_pos = self.agent_state[0] * 4 + self.agent_state[1]
            legal_actions_indecies = self.generate_legal_actions(self.agent_state)
            print("Legal Actions:",legal_actions_indecies)

            # choosing max from only legal moves 
            legal_q_values = []
            for i in range(len(legal_actions_indecies)):
                legal_q_values.append(self.q_table[agent_pos][legal_actions_indecies[i]])

            print("Q_VALUES:",self.q_table[agent_pos])
            print("LEGAL Q_VALUES:",legal_q_values)

            raw_action = np.argmax(legal_q_values)
            print("RAW ACTION:",raw_action)

            action = self.all_actions[legal_actions_indecies[raw_action]]
            print("ACTION",action)
            return action
    

    def excecute_move(self,action):
        #main updates
        print("ACTION:",action)
        reward = self.reward(self.agent_state,action)
        print("Reward:",reward)
        next_pos = self.agent_state[0] + action[0], self.agent_state[1] + action[1]
        next_state = next_pos[0] * 4 + next_pos[1]
        print("Next State:",next_state)
        
        self.state_counter[next_pos[0]][next_pos[1]] += 1 

        #show on the board

        self.Maze[self.agent_state[0]][self.agent_state[1]] = 1
        self.Maze[next_pos[0]][next_pos[1]] = 'S'

        return next_state,reward
    
    def main_Qlearning(self):
        # returns actual action (0,1) to pass to update we need to turn into indicies
        action = self.generate_move()
        # action to action index --> to update q_value
        for i in range(len(self.all_actions)):
            if action == self.all_actions[i]:
                action_index = i
        
        next_state,reward = self.excecute_move(action)

        self.update_q(self.agent,action_index,reward,next_state)

        self.agent = next_state
        self.agent_state = self.agent // len(self.Maze), self.agent % len(self.Maze[0])

        if self.episode_over == True:
            self.episode_counter += 1
            # epsilon decay
            self.EPSILON = self.EPSILON * 0.9
            self.print_info()
            self.reset()

    def reset(self):
        self.Maze = [['S',0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.agent = 0
        self.agent_state = (0,0)
        self.episode_over = False
        self.wrong_move = False
    
    def print_boards(self,board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                print(board[i][j],end=" ")
            print()
    
    def print_info(self):
        print("EPISODE OVER!!!!!!")
        print("EPISODE COUNTER:",self.episode_counter)
        print("-----------------------------")
        self.print_boards(self.Maze)
        print("--------------------------")
        self.print_boards(self.state_counter)
        print(self.q_table)
        print("-----------------------------")






