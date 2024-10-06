from flask import Flask, jsonify, request, render_template
import torch
from environment import TicTacToeEnv
from deep_q_networks import DQN

app = Flask(__name__)

# Initialize the game environment
env = TicTacToeEnv()

# Load the trained agent
agent = DQN(9, 9)
agent.load_state_dict(torch.load('saved_model/model_weights.pth'))
agent.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/init', methods=['GET'])
def init_game():
    state = env.reset()
    return jsonify({'board': env.board.flatten().tolist()})

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    move = data['move']
    state, reward, done, _ = env.step(-1, move)
    
    if done:
        return jsonify({'board': env.board.flatten().tolist(), 'done': True, 'winner': 'human' if reward == 1 else 'draw'})

    with torch.no_grad():
        q_values = agent(torch.FloatTensor(state))
        valid_actions = [i for i in range(9) if env.board.flatten()[i] == 0]
        action = valid_actions[torch.argmax(q_values[valid_actions]).item()]
        state, reward, done, _ = env.step(1, action)

    return jsonify({'board': env.board.flatten().tolist(), 'done': done, 'winner': 'agent' if reward == 1 else 'draw'})

if __name__ == '__main__':
    app.run(debug=True)
