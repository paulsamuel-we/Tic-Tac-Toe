from flask import Flask, jsonify, request, render_template
import torch
import numpy as np
from environment import TicTacToeEnv
from deep_q_networks import DQN

app = Flask(__name__)

# Initialize the game environment
env = TicTacToeEnv()

# Load the trained bot (assume model is saved as 'bot.pth')
bot = DQN(9, 9)
bot.load_state_dict(torch.load('saved_model/model_weights.pth'))
bot.eval()

# Route to get the initial board state
@app.route('/init', methods=['GET'])
def init_game():
    state = env.reset()  # Reset environment
    return jsonify({'board': env.board.tolist()})

# Route to make a move by human and get the bot's response
@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    move = data['move']  # Human's move
    state, reward, done, _ = env.step(-1, move)  # Human makes a move
    
    if done:
        return jsonify({'board': env.board.tolist(), 'done': True, 'winner': 'human' if reward == 1 else 'draw'})
    
    # Agent's move
    with torch.no_grad():
        q_values = bot(torch.FloatTensor(state))
        valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
        agent_move = valid_actions[torch.argmax(q_values[valid_actions]).item()]
        state, reward, done, _ = env.step(1, agent_move)

    if done:
        return jsonify({'board': env.board.tolist(), 'done': True, 'winner': 'bot' if reward == 1 else 'draw'})
    
    return jsonify({'board': env.board.tolist(), 'done': False})

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
