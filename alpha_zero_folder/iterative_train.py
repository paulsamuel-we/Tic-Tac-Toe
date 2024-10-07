import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import TicTacToeEnv
from mcts import MCTS
from model import PolicyValueNetwork  # Ensure this is your model class

def train_alpha_zero(episodes=10000, num_simulations=100, save_path='saved_model/new_model.pth', saved_model_path=None):
    env = TicTacToeEnv()

    # Load the saved model, or initialize a new one
    if saved_model_path:
        saved_model = PolicyValueNetwork()  # Initialize saved model
        saved_model.load_state_dict(torch.load(saved_model_path))
        saved_model.eval()  # Ensure it's in eval mode
    else:
        saved_model = PolicyValueNetwork()  # New model for self-play

    # Initialize the new model that will learn
    new_model = PolicyValueNetwork()

    # Set different learning rates for policy head and value head of the new model
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

    # Set up MCTS for both models
    mcts_new = MCTS(new_model, num_simulations)
    mcts_saved = MCTS(saved_model, num_simulations)

    for episode in range(episodes):
        # Randomly choose whether the new model plays against itself or the saved model
        if np.random.rand() < 0.5:
            model_1, model_2 = new_model, new_model  # Self-play (new model vs itself)
        else:
            model_1, model_2 = new_model, saved_model  # Play against saved model
        
        # Reset the game environment
        state = env.reset()
        visit_counts = np.zeros(9)
        rewards = []
        current_model = model_1

        # Play the game using MCTS and alternate between models
        while not env.done:
            root = MCTS(current_model, num_simulations).get_root(state)
            MCTS(current_model, num_simulations).run(root)
            action = MCTS(current_model, num_simulations).select_action(root)

            next_state, reward, done, _ = env.step(env.agent_symbol, action)
            visit_counts[action] += 1
            rewards.append(reward)

            state = next_state
            current_model = model_2 if current_model == model_1 else model_1

        # Calculate target policy and value
        target_policy = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else visit_counts
        target_value = sum(rewards)

        # Prepare data for training
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        target_policy_tensor = torch.FloatTensor(target_policy).unsqueeze(0)
        target_value_tensor = torch.FloatTensor([target_value]).unsqueeze(0)

        # Forward pass through the new model
        optimizer.zero_grad()
        policy, value = new_model(state_tensor)

        # Calculate the combined policy and value loss
        policy_loss = -torch.sum(target_policy_tensor * torch.log(policy + 1e-10))  # Adding small value to prevent log(0)
        value_loss = torch.nn.functional.mse_loss(value, target_value_tensor)
        loss = policy_loss + value_loss

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        # Logging (optional)
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Loss: {loss.item()}, Value Loss: {value_loss.item()}, Policy Loss: {policy_loss.item()}")

    # Save the new model
    torch.save(new_model.state_dict(), save_path)

    return new_model

# Function to evaluate models after training
def evaluate_models(new_model, saved_model, num_games=100):
    env = TicTacToeEnv()
    new_model_wins, saved_model_wins = 0, 0

    for game in range(num_games):
        state = env.reset()
        current_model = new_model if game % 2 == 0 else saved_model

        # Play the game with alternating turns
        while not env.done:
            root = MCTS(current_model, num_simulations=100).get_root(state)
            MCTS(current_model, num_simulations=100).run(root)
            action = MCTS(current_model, num_simulations=100).select_action(root)
            state, _, done, _ = env.step(env.agent_symbol, action)
            current_model = saved_model if current_model == new_model else new_model

        # Count wins
        if env.winner == 1:  # Assuming 1 represents the new model's symbol
            new_model_wins += 1
        else:
            saved_model_wins += 1

    print(f"New Model Wins: {new_model_wins}, Saved Model Wins: {saved_model_wins}")

# Main function to run training and evaluation
if __name__ == "__main__":
    # Train the new model, optionally loading an old saved model
    model = train_alpha_zero(episodes=10000, saved_model_path='saved_model/policy_model_weights.pth')

    # Load the saved model to play against the new model
    saved_model = PolicyValueNetwork()
    saved_model.load_state_dict(torch.load('saved_model/policy_model_weights.pth'))
    saved_model.eval()

    # Evaluate the performance of the new model against the saved model
    evaluate_models(model, saved_model, num_games=100)