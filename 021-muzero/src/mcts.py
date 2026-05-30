import math
import numpy as np


class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.hidden_state = None
        self.reward = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def ucb_score(parent: Node, child: Node, c_puct: float = 1.25) -> float:
    exploration = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return child.value + exploration


def expand(node: Node, policy_logits, hidden_state, reward: float, n_actions: int) -> None:
    node.hidden_state = hidden_state
    node.reward = reward
    probs = softmax(policy_logits)
    for a in range(n_actions):
        node.children[a] = Node(prior=float(probs[a]))


def softmax(logits) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def backpropagate(path: list[Node], value: float, discount: float) -> None:
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward + discount * value


def run_mcts(root: Node, networks, n_simulations: int, n_actions: int,
             discount: float = 0.997) -> np.ndarray:
    import torch
    dynamics_net, prediction_net = networks

    for _ in range(n_simulations):
        node = root
        path = [node]

        # Selection: follow UCB until unexpanded node
        while node.is_expanded():
            action = max(node.children, key=lambda a: ucb_score(node, node.children[a]))
            node = node.children[action]
            path.append(node)

        # Expansion: use dynamics to get next hidden state + reward
        parent = path[-2]
        action = [a for a, child in parent.children.items() if child is node][0]
        a_onehot = torch.zeros(1, n_actions)
        a_onehot[0, action] = 1.0
        hidden = torch.FloatTensor(parent.hidden_state).unsqueeze(0)

        with torch.no_grad():
            next_hidden, reward = dynamics_net(hidden, a_onehot)
            policy_logits, value = prediction_net(next_hidden)

        expand(node, policy_logits[0], next_hidden[0].numpy(), reward.item(), n_actions)

        # Backpropagation
        backpropagate(path, value.item(), discount)

    # Action selection: proportional to visit counts
    visits = np.array([root.children[a].visit_count for a in range(n_actions)], dtype=np.float64)
    return visits / visits.sum()
