import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from deep_q_networks import DQN
from environment import TicTacToeEnv


def train_double_dqn(env, main_network, target_network, episodes=1000, update_target_every=10):
    optimizer = optim.Adam(main_network.parameters())
    criterion = nn.MSELoss()
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = main_network(torch.FloatTensor(state))
                action = valid_actions[torch.argmax(q_values[valid_actions]).item()]

            next_state, reward, done, _ = env.step(1, action)
            total_reward += reward

            if not done:
                # Human takes a random action (or you can implement human logic)
                valid_human_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
                if valid_human_actions:
                    human_action = random.choice(valid_human_actions)
                    _, human_reward, done, _ = env.step(-1, human_action)

            # Compute target using the Double DQN formula
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

        # Update epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Update target network periodically
        if episode % update_target_every == 0:
            target_network.load_state_dict(main_network.state_dict())
        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')




# //---------------------------------------------------------------------------
# Now lets train our DQN with dueling approach
env = TicTacToeEnv()
main_network = DQN(9, 9)
target_network = DQN(9, 9)

train_double_dqn(env, main_network, target_network, episodes=10000)


# //---------------------------------------------------------------------------
# save the model

torch.save(main_network, 'saved_model/model.pth')

# save the model's state_dict
torch.save(main_network.state_dict(), 'saved_model/model_weights.pth')