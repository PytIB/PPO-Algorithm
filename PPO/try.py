import random

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[0 for i in range(width)] for j in range(height)]
        self.start_position = (0, 0)
        self.end_position = (width - 1, height - 1)
        self.walls = []

    def generate_maze(self):
        for i in range(self.width):
            for j in range(self.height):
                if i == 0 or i == self.width - 1 or j == 0 or j == self.height - 1:
                    self.maze[i][j] = 1
                elif random.random() < 0.5:
                    self.maze[i][j] = 1
                    self.walls.append((i, j))

    def is_valid_position(self, position):
        i, j = position
        if i < 0 or i >= self.width or j < 0 or j >= self.height:
            return False
        if self.maze[i][j] == 1:
            return False
        return True

    def get_neighbors(self, position):
        i, j = position
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for neighbor in neighbors:
            if self.is_valid_position(neighbor):
                yield neighbor

    def solve(self):
        policy = PPOPolicy(self.width, self.height)
        for _ in range(100000):
            policy.reset()
            while not policy.is_done():
                action = policy.select_action()
                new_position = policy.position + action
                if self.is_valid_position(new_position):
                    policy.position = new_position
                    policy.reward = 1 if new_position == self.end_position else 0
                    policy.update()
        return policy.position

class PPOPolicy:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.position = (0, 0)
        self.reward = 0
        self.done = False
        self.memory = []
        self.gamma = 0.99
        self.alpha = 0.0001
        self.epsilon = 0.1
        self.v = [[0 for i in range(width)] for j in range(height)]
        self.end_position = (width - 1, height - 1)

    def get_neighbors(self, position):
        i, j = position
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                yield list(neighbor)

    def select_action(self):
        if random.random() < self.epsilon:
            return random.choice(tuple(self.get_neighbors(self.position)))
        else:
            q_values = []
            for action in self.get_neighbors(self.position):
                q_values.append(self.get_q_value(action))
            return list(action)[self.get_best_action(q_values)]

    def get_best_action(self, q_values):
        return q_values.index(max(q_values))

    def get_q_value(self, action):
        if self.is_done():
            return 0
        next_position = (self.position[0] + action[0], self.position[1] + action[1])
        return self.v[next_position[0]][next_position[1]]

    def update(self):
        state = (self.position, self.reward)
        self.memory.append(state)
        for _ in range(10):
            s, a, r, s_prime = self.memory.pop(0)
            td_error = r + self.gamma * self.get_q_value(s_prime) - self.get_q_value(a)
            self.v[s[0]][s[1]] += self.alpha * td_error

    def is_done(self):
        return self.position == self.end_position

    def solve(self):
        while not self.is_done():
            action = self.select_action()
            new_position = list(self.position) + list(action)
            if self.is_valid_position(new_position):
                self.position = new_position
                self.reward = 1 if new_position == self.end_position else 0
                self.update()
        return self.position


def main():
    maze = Maze(4, 4)
    maze.generate_maze()
    policy = PPOPolicy(4, 4)
    policy.solve()
    print(policy.position)

if __name__ == "__main__":
    main()