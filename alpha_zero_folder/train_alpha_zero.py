import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import TicTacToeEnv
from mcts import MCTS
from model import PolicyValueNetwork

def train_alpha_zero(episodes=10000, num_simulations=100):
    env = TicTacToeEnv()
    policy_value_net = PolicyValueNetwork()
    mcts = MCTS(policy_value_net, num_simulations)
    optimizer = torch.optim.Adam(policy_value_net.parameters(), lr=0.001)

    for episode in range(episodes):
        state = env.reset()
        visit_counts = np.zeros(9)
        rewards = []

        while not env.done:
            # Use MCTS to find the best action
            root = mcts.get_root(state)
            mcts.run(root)
            action = mcts.select_action(root)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(env.agent_symbol, action)
            visit_counts[action] += 1
            rewards.append(reward)

            # Update state
            state = next_state

        # Calculate target policy and value
        target_policy = visit_counts / np.sum(visit_counts)
        target_value = sum(rewards)

        # Prepare data for training
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        target_policy_tensor = torch.FloatTensor(target_policy).unsqueeze(0)
        target_value_tensor = torch.FloatTensor([target_value]).unsqueeze(0)

        # Forward pass through the network
        optimizer.zero_grad()
        policy, value = policy_value_net(state_tensor)

        # Calculate loss (combined policy and value loss)
        policy_loss = torch.sum(-target_policy_tensor * torch.log(policy))
        value_loss = torch.nn.functional.mse_loss(value, target_value_tensor)
        loss = policy_loss + value_loss

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        # Logging (optional)
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Loss: {loss.item()}, Value Loss: {value_loss.item()}, Policy Loss: {policy_loss.item()}")

    return policy_value_net
    

# Start training
if __name__ == "__main__":
    model = train_alpha_zero(episodes=10000)

    # save the model
    torch.save(model, 'saved_model/policy_model.pth')
    torch.save(model.state_dict(), 'saved_model/policy_model_weights.pth')