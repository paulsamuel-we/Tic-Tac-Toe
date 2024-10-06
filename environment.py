import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 3x3 grid, 0 for empty, 1 for agent, -1 for human
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3))
        self.done = False
        self.winner = None
        return self.board.flatten()

    def check_winner(self):
        for i in range(3):
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                self.done = True
                self.winner = 1
                return 1
            if np.all(self.board[i, :] == -1) or np.all(self.board[:, i] == -1):
                self.done = True
                self.winner = -1
                return -1
        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            self.done = True
            self.winner = 1
            return 1
        if np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1):
            self.done = True
            self.winner = -1
            return -1
        if not np.any(self.board == 0):  # No more empty spaces, it's a draw
            self.done = True
            self.winner = 0
            return 0
        return None

    def step(self, player, action):
        if self.done or self.board[action // 3, action % 3] != 0:
            return self.board.flatten(), -10, self.done, {}
        self.board[action // 3, action % 3] = player
        reward = 0
        result = self.check_winner()
        if result == 1:
            reward = 10
        elif result == -1:
            reward = -10
        elif result == 0:
            reward = 0
        return self.board.flatten(), reward, self.done, {}
