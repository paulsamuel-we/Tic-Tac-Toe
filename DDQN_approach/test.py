import torch
from environment import TicTacToeEnv
from deep_q_networks import DQN  # Make sure to import your DQN class

def human_vs_agent(env, agent_path):
    # Load the agent model
    agent = DQN(9, 9)  # Adjust input and output sizes based on your model
    agent.load_state_dict(torch.load(agent_path))
    agent.eval()  # Set the agent to evaluation mode

    state = env.reset()
    done = False
    print(env.board)  # Print initial state of the board
    
    while not done:
        # Human turn
        valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
        if not valid_actions:
            print("No more moves left! It's a draw.")
            break
        
        move = int(input("Enter your move (0-8): "))
        while move not in valid_actions:
            print("Invalid move. Please choose a valid move.")
            move = int(input("Enter your move (0-8): "))

        state, reward, done, _ = env.step(-1, move)  # Human move
        print(env.board)  # Print board after human's move

        if done:
            if reward == 1:
                print("Human wins!")
            else:
                print("It's a draw!")
            break

        # Agent turn
        with torch.no_grad():
            q_values = agent(torch.FloatTensor(state))
            valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
            if not valid_actions:
                print("No more moves left! It's a draw.")
                break
            
            action = valid_actions[torch.argmax(q_values[valid_actions]).item()]
            state, reward, done, _ = env.step(1, action)  # Agent move
            print(env.board)  # Print board after agent's move

        if done:
            if reward == 1:
                print("Agent wins!")
            else:
                print("It's a draw!")
            break


# Let's test it
human_vs_agent(env=TicTacToeEnv(), agent_path='saved_model/model_weights.pth')