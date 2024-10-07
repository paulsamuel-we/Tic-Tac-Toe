import torch
from environment import TicTacToeEnv
from mcts import MCTS
from model import PolicyValueNetwork

def human_vs_agent(model_path, num_simulations=100):
    # Load the pre-trained model
    model = PolicyValueNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    env = TicTacToeEnv()
    done = False

    print("Initial Board:")
    print(env.board)

    while not done:
        # Human turn
        valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
        move = int(input("Enter your move (0-8): "))
        while move not in valid_actions:
            move = int(input("Invalid move. Enter a valid move (0-8): "))
        
        state, _, done, _ = env.step(-1, move)  # Update the state with the human's move
        print(env.board)

        if done:
            print("Human wins!")
            break

        # Agent's turn
        root = MCTS(model, num_simulations).get_root(env.board.flatten())  # Using the flattened board as state
        MCTS(model, num_simulations).run(root)
        action = MCTS(model, num_simulations).select_action(root)

        # Take the action in the environment
        state, _, done, _ = env.step(1, action)  # Update the state with the agent's move
        print(env.board)

        if done:
            print("Agent wins!")
            break


# Main function to run the human vs. model game
if __name__ == "__main__":
    model_path = 'saved_model/new_model_weights.pth'  # Path to your trained model
    human_vs_agent(model_path)
