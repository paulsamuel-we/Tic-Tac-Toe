from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from environment import TicTacToeEnv
from DDQN_approach.deep_q_networks import DQN

app = Flask(__name__)

# Load the trained model
model_path = 'saved_model/policy_model_weights.pth'
model = DQN(9, 9)
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize TicTacToe environment
env = TicTacToeEnv()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    board = np.array(data['board']).reshape(3, 3)
    player_symbol = 1 if data['player_symbol'] == 'X' else -1

    # Update environment with the current board state
    env.board = board

    # Check if the game is over before AI makes a move
    result = env.check_winner()
    if result == player_symbol:
        return jsonify(status='game_over', message='You win!', board=board.flatten().tolist())
    elif result == -player_symbol:
        return jsonify(status='game_over', message='AI wins!', board=board.flatten().tolist())
    elif result == 0:
        return jsonify(status='game_over', message='It\'s a draw!', board=board.flatten().tolist())

    # AI makes its move
    with torch.no_grad():
        q_values = model(torch.FloatTensor(env.board.flatten()))
        valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
        ai_action = valid_actions[torch.argmax(q_values[valid_actions]).item()]

    env.step(1, ai_action)  # AI plays as 'X'

    # Check game status again after AI's move
    result = env.check_winner()
    if result == player_symbol:
        return jsonify(status='game_over', message='You win!', board=env.board.flatten().tolist())
    elif result == -player_symbol:
        return jsonify(status='game_over', message='AI wins!', board=env.board.flatten().tolist())
    elif result == 0:
        return jsonify(status='game_over', message='It\'s a draw!', board=env.board.flatten().tolist())

    return jsonify(status='continue', board=env.board.flatten().tolist())

if __name__ == '__main__':
    app.run(debug=True)
