import pygame



class Maze:
    def __init__(self,width,heigh) -> None:
       

        self.WIDTH = width
        self.HEIGHT = heigh
        self.BLACK = (64,64,64)
        self.REAL_BLACK = (0,0,0)
        self.RED = (220,20,60)
        self.YELLOW = (184,134,11)
        self.GREEN = (0,128,0)
        self.Clock = pygame.time.Clock()
        self.episode_counter = 100
        #self.t1 = threading.Thread(target=self.solver_Delay)
        pygame.init()
        pygame.display.set_caption("RL Maze")
        self.SCREEN = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
    def display(self):
        #self.display_maze(myMaze)
        pygame.time.wait(100)
        #self.myPPO()
    def game_loop(self):
        self.SCREEN.fill('black')
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            self.display()
            pygame.display.update()
            self.Clock.tick(60)

    def display_maze(self,Maze):
        rows = len(Maze)
        columns = len(Maze[0]) 
        margin = 2
        block_size = self.WIDTH / rows - margin
        full_width, full_height = margin + block_size, margin + block_size  
        for i in range(rows):
            for j in range(columns):
                if Maze[i][j] == 0:
                    pygame.draw.rect(self.SCREEN, self.BLACK,[full_width * j +margin, 
                                            full_height * i +margin, block_size, block_size])
                if Maze[i][j] == 'S':
                    pygame.draw.rect(self.SCREEN, self.RED,[full_width * j +margin, 
                                            full_height * i +margin, block_size, block_size])
                    
                if Maze[i][j] == 'F':
                    pygame.draw.rect(self.SCREEN, self.GREEN,[full_width * j +margin, 
                                            full_height * i +margin, block_size, block_size])
                if i == 1 and j == 2:
                    pygame.draw.rect(self.SCREEN, ((25,25,112)),[full_width * j +margin, 
                                            full_height * i +margin, block_size, block_size])
                if Maze[i][j] == 1:
                    pygame.draw.rect(self.SCREEN, self.YELLOW,[full_width * j +margin, 
                                            full_height * i +margin, block_size, block_size])
     


if __name__ == "__main__":
    #mymaze = [['S',0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,'F']]
    maze = Maze(800,800)
    #maze.get_maze()
   # maze.t1.start()
    maze.game_loop()
    