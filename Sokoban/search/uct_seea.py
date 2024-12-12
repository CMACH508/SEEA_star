import numpy as np
from memory import Trajectory
import copy
import time


class MinMaxStats(object):
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class AStarTreeNode:
    def __init__(self, parent, game_state, g, f, action, cp):
        self._game_state = game_state
        self._g = g
        self._f = f
        self._action = action
        self._parent = parent
        if parent is None:
            self._d = 1
        else:
            self._d = self._parent._d + 1
        self._cp = cp

    def __eq__(self, other):
        return self._game_state == other._game_state

    def __lt__(self, other):
        return self._f < other._f

    def __hash__(self):
        return self._game_state.__hash__()

    def get_g(self):
        return self._g

    def get_f(self):
        return self._f

    def get_game_state(self):
        return self._game_state

    def get_parent(self):
        return self._parent

    def get_action(self):
        return self._action

    def set_f_cost(self, f_cost):
        self._f = f_cost

    def get_stat(self, max_d, min_max_stats_f):
        return min_max_stats_f.normalize(self._f) - self._cp * np.sqrt(max_d) / (1 + self._d)


class SeeA():
    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansion=32, weight=1.0, candidate_size=100, cp=0.2):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._w = weight
        self._k = k_expansion
        self._candidate_size = candidate_size
        self._cp = cp

    def get_f_cost(self, child, g, predicted_h):
        if self._use_learned_heuristic and self._use_heuristic:
            return self._w * max(predicted_h, child.heuristic_value()) + g
        if self._use_learned_heuristic:
            return self._w * predicted_h + g
        if self._use_heuristic:
            return self._w * child.heuristic_value() + g
        return g

    def store_trajectory_memory(self, tree_node, expanded):
        states = []
        actions = []
        solution_costs = []
        state = tree_node.get_parent()
        action = tree_node.get_action()
        cost = 1
        while not state.get_parent() is None:
            states.append(state.get_game_state())
            actions.append(action)
            solution_costs.append(cost)
            action = state.get_action()
            state = state.get_parent()
            cost += 1
        states.append(state.get_game_state())
        actions.append(action)
        solution_costs.append(cost)
        return Trajectory(states, actions, solution_costs, expanded)

    def search(self, data):
        state, puzzle_name, nn_model, budget, start_overall_time, time_limit, slack_time = data
        start_time = time.time()
        if slack_time == 0:
            start_overall_time = time.time()
        root = AStarTreeNode(None, state, 0, 0, -1, self._cp)
        _open = [root]
        min_max_stats_f = MinMaxStats()
        min_max_stats_f.update(root._f)
        _closed = set()
        _closed.add(state)
        expanded, generated = 0, 0
        predicted_h = np.zeros(self._k * 2)
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []
        while len(_open) > 0:
            if len(_open) <= self._candidate_size:
                f_value = [node.get_f() for node in _open]
                expanded_idx = np.argmin(f_value)
            else:
                max_d = np.max([node._d for node in _open])
                stats = [node.get_stat(max_d, min_max_stats_f) for node in _open]
                index_candidate = np.argsort(stats)[: self._candidate_size]
                stats = [_open[i].get_f() for i in index_candidate]
                expanded_idx = index_candidate[np.argmin(stats)]
            node = _open[expanded_idx]
            _open = _open[: expanded_idx] + _open[expanded_idx + 1:]
            expanded += 1
            end_time = time.time()
            if (budget > 0 and expanded > budget) or end_time - start_overall_time + slack_time > time_limit:
                return -1, expanded, generated, end_time - start_time, puzzle_name
            actions = node.get_game_state().successors_parent_pruning(node.get_action())
            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)
                generated += 1
                if child.is_solution():
                    end_time = time.time()
                    return node.get_g() + 1, expanded, generated, end_time - start_time, puzzle_name
                child_node = AStarTreeNode(node, child, node.get_g() + 1, -1, a, self._cp)
                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
            if self._use_learned_heuristic:
                predicted_h = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                predicted_h = predicted_h.reshape(-1)
            for i in range(len(children_to_be_evaluated)):
                f_cost = self.get_f_cost(children_to_be_evaluated[i].get_game_state(),
                                         children_to_be_evaluated[i].get_g(), predicted_h[i])
                children_to_be_evaluated[i].set_f_cost(f_cost)
                if children_to_be_evaluated[i].get_game_state() not in _closed:
                    _open.append(children_to_be_evaluated[i])
                    min_max_stats_f.update(children_to_be_evaluated[i]._f)
                    _closed.add(children_to_be_evaluated[i].get_game_state())
            children_to_be_evaluated.clear()
            x_input_of_children_to_be_evaluated.clear()