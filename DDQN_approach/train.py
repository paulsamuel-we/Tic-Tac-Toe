import random
import torch
import torch.optim as optim
import torch.nn as nn
from deep_q_networks import DQN
from environment import TicTacToeEnv


def train_double_dqn(env, main_network, target_network, episodes=50000, update_target_every=5):
    optimizer = optim.Adam(main_network.parameters())
    criterion = nn.MSELoss()
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999  # Slower decay to ensure more exploration early on
    epsilon_min = 0.1  # Let epsilon drop a bit lower to increase difficulty later

    for episode in range(episodes):
        # Randomly assign the agent to play as X (1) or O (-1)
        agent_symbol = random.choice([1, -1])
        state = env.reset(agent_symbol)
        done = False
        total_reward = 0
        human_penalty = -5  # Penalize the agent for allowing human to make a good move

        while not done:
            valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]

            # Epsilon-greedy action selection with slower decay
            if random.uniform(0, 1) < epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = main_network(torch.FloatTensor(state))
                action = valid_actions[torch.argmax(q_values[valid_actions]).item()]

            # Agent makes a move
            next_state, reward, done, _ = env.step(agent_symbol, action)
            total_reward += reward

            if not done:
                # Human takes a random action (or replace this with a minimax-based AI or stronger random logic)
                valid_human_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
                if valid_human_actions:
                    human_action = random.choice(valid_human_actions)
                    _, human_reward, done, _ = env.step(-agent_symbol, human_action)
                    total_reward += human_penalty if human_reward > 0 else 0  # Penalize agent for letting human make a good move

            # Compute target using Double DQN
            q_values = main_network(torch.FloatTensor(state))
            next_q_values_main = main_network(torch.FloatTensor(next_state))
            next_q_values_target = target_network(torch.FloatTensor(next_state))

            next_action = torch.argmax(next_q_values_main).item()
            target = reward + gamma * next_q_values_target[next_action].item() * (not done)

            # Update main network
            loss = criterion(q_values[action], torch.FloatTensor([target]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        # Update epsilon more slowly and allow it to drop lower for exploration
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Update target network more frequently for quicker learning
        if episode % update_target_every == 0:
            target_network.load_state_dict(main_network.state_dict())

        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')


# Initialize environment and networks
env = TicTacToeEnv()
main_network = DQN(9, 9)
target_network = DQN(9, 9)

train_double_dqn(env, main_network, target_network, episodes=10000)

# Save the model
torch.save(main_network, 'saved_model/model.pth')
torch.save(main_network.state_dict(), 'saved_model/model_weights.pth')
