import numpy as np
import torch

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state.copy()  # Ensure we are working with a copy of the state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.action = action
        self.is_expanded = False

    def expand(self, valid_actions):
        """ Expands the node by creating children nodes for all valid actions """
        for action in valid_actions:
            child_state = self.state.copy()
            # Alternate between X and O moves
            player_symbol = 1 if (np.count_nonzero(self.state) % 2 == 0) else -1
            child_state[action] = player_symbol
            self.children.append(Node(child_state, parent=self, action=action))
        self.is_expanded = True

    def get_value(self):
        return self.value_sum / (self.visit_count + 1e-5)  # To avoid division by zero


class MCTS:
    def __init__(self, network, num_simulations):
        self.network = network
        self.num_simulations = num_simulations
        self.root = None

    def get_root(self, state):
        """ Creates a new root node based on the current state. """
        self.root = Node(state)
        return self.root

    def run(self, root):
        """ Run MCTS simulations starting from the root node """
        for _ in range(self.num_simulations):
            node = root
            while node.is_expanded:
                node = self._select_child(node)  # Select the best child

            # Expand the node (generate children nodes)
            valid_actions = self._get_valid_actions(node.state)
            if valid_actions:
                node.expand(valid_actions)
                self._evaluate_leaf(node)

    def _select_child(self, node):
        """ Use UCT to select the child with the highest score """
        best_score = -np.inf
        best_child = None
        c_puct = 1.7  # Exploration constant, can be tuned

        for child in node.children:
            uct_score = child.get_value() + c_puct * np.sqrt(node.visit_count) / (1 + child.visit_count)
            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def _get_valid_actions(self, state):
        """ Returns a list of valid actions (empty positions on the board) """
        return [i for i, val in enumerate(state.flatten()) if val == 0]

    def _evaluate_leaf(self, node):
        """ Evaluate the leaf node using the neural network """
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
        policy, value = self.network(state_tensor)
        node.value_sum = value.item()
        node.visit_count += 1

    def select_action(self, root):
        """ Select the action based on the visit counts of the children """
        visit_counts = [child.visit_count for child in root.children]
        if len(visit_counts) == 0:
            raise ValueError("No children nodes to select from.")
        return root.children[np.argmax(visit_counts)].action
