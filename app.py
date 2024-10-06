from flask import Flask, jsonify, request, render_template
import torch
import numpy as np

app = Flask(__name__)

# Assuming you have already defined the TicTacToeEnv and DQN classes
from the_game_module import TicTacToeEnv, DQN

# Initialize the game environment
env = TicTacToeEnv()

# Load the trained agent (assume model is saved as 'agent.pth')
agent = DQN(9, 9)
agent.load_state_dict(torch.load('model\bot.pth'))
agent.eval()

# Route to get the initial board state
@app.route('/init', methods=['GET'])
def init_game():
    state = env.reset()  # Reset environment
    return jsonify({'board': env.board.tolist()})

# Route to make a move by human and get the agent's response
@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    move = data['move']  # Human's move
    state, reward, done, _ = env.step(-1, move)  # Human makes a move
    
    if done:
        return jsonify({'board': env.board.tolist(), 'done': True, 'winner': 'human' if reward == 1 else 'draw'})
    
    # Agent's move
    with torch.no_grad():
        q_values = agent(torch.FloatTensor(state))
        valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
        agent_move = valid_actions[torch.argmax(q_values[valid_actions]).item()]
        state, reward, done, _ = env.step(1, agent_move)

    if done:
        return jsonify({'board': env.board.tolist(), 'done': True, 'winner': 'agent' if reward == 1 else 'draw'})
    
    return jsonify({'board': env.board.tolist(), 'done': False})

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
