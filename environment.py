import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 3x3 grid, 0 for empty, 1 for X, -1 for O
        self.done = False
        self.winner = None
        self.agent_symbol = 1  # Agent plays as X by default
        self.human_symbol = -1  # Human plays as O by default

    def reset(self, agent_symbol=1):
        """ Reset the board for a new game, optionally setting agent symbol (X or O) """
        self.board = np.zeros((3, 3))
        self.done = False
        self.winner = None
        self.agent_symbol = agent_symbol
        self.human_symbol = -agent_symbol  # Ensure the human gets the opposite symbol
        return self.board.flatten()

    def check_winner(self):
        # Check rows and columns for a win
        for i in range(3):
            if np.all(self.board[i, :] == self.agent_symbol) or np.all(self.board[:, i] == self.agent_symbol):
                self.done = True
                self.winner = self.agent_symbol
                return self.agent_symbol  # Return agent win
            if np.all(self.board[i, :] == self.human_symbol) or np.all(self.board[:, i] == self.human_symbol):
                self.done = True
                self.winner = self.human_symbol
                return self.human_symbol  # Return human win

        # Check diagonals for a win
        if np.all(np.diag(self.board) == self.agent_symbol) or np.all(np.diag(np.fliplr(self.board)) == self.agent_symbol):
            self.done = True
            self.winner = self.agent_symbol
            return self.agent_symbol  # Agent win
        if np.all(np.diag(self.board) == self.human_symbol) or np.all(np.diag(np.fliplr(self.board)) == self.human_symbol):
            self.done = True
            self.winner = self.human_symbol
            return self.human_symbol  # Human win

        # Check for draw (if no zeros are left and no winner)
        if not np.any(self.board == 0):  # No more empty spaces
            self.done = True
            self.winner = 0
            return 0  # Draw

        return None

    def step(self, player, action):
        # Validate the move (return penalty for invalid move)
        if self.done or self.board[action // 3, action % 3] != 0:
            return self.board.flatten(), -10, self.done, {}  # Invalid move penalty

        # Make the move
        self.board[action // 3, action % 3] = player

        # Check for a win/draw
        result = self.check_winner()

        # Set reward based on the result
        if result == player:  # Player wins
            reward = 1  # Winning reward
        elif result == -player:  # Opponent wins
            reward = -1  # Losing penalty
        elif result == 0:  # Draw
            reward = 0  # Small reward for draw
        else:
            reward = 0  # No result yet, neutral reward

        return self.board.flatten(), reward, self.done, {}
