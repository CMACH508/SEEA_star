import numpy as np
from memory import Trajectory
import copy
import time
from prepare import prepare_data
from valueModel import PVModel


class AStarTreeNode:
    def __init__(self, parent, game_state, g, f, action, cluster_ID=0):
        self._game_state = game_state
        self._g = g
        self._f = f
        self._action = action
        self._parent = parent
        self.cluster_ID = cluster_ID

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


class SeeA():
    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansion=32, weight=1.0, candidate_size=100, num_clusters=5, c_weight=0.1):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._w = weight
        self._k = k_expansion
        self._candidate_size = candidate_size
        self._num_clusters = num_clusters
        self._c_weight = c_weight

    def init_cluster_center(self):
        centers = [np.random.rand(100) for _ in range(self._num_clusters)]
        return np.array(centers)

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
        cluster_centers = self.init_cluster_center()
        if slack_time == 0:
            start_overall_time = time.time()
        root = AStarTreeNode(None, state, 0, 0, -1)
        x_input_of_children_to_be_evaluated = [root._game_state.get_image_representation()]
        predicted_h, predicted_feature = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
        f_cost = self.get_f_cost(root.get_game_state(), root.get_g(), predicted_h[0])
        root.set_f_cost(f_cost)
        distance = np.linalg.norm(cluster_centers - predicted_feature[0], axis=1)
        cluster_ID = np.argmin(distance)
        root.cluster_ID = cluster_ID
        cluster_centers[root.cluster_ID] = (1 - self._c_weight) * cluster_centers[root.cluster_ID] + self._c_weight * predicted_feature[0]
        _open = [[] for _ in range(self._num_clusters)]
        _open[root.cluster_ID].append(root)
        open_size = 1
        _closed = set()
        _closed.add(state)
        expanded, generated = 0, 0
        predicted_h = np.zeros(self._k * 2)
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []
        while open_size > 0:
            index_cluster, index_node, value_node = [], [], []
            num_nodes_per_cluster = int(self._candidate_size // self._num_clusters)
            for i in range(self._num_clusters):
                if len(_open[i]) > 0:
                    index_cluster.append(i)
                    if len(_open[i]) <= num_nodes_per_cluster:
                        cluster_values = [_open[i][j]._f for j in range(len(_open[i]))]
                        index_node.append(np.argmin(cluster_values))
                        value_node.append(cluster_values[index_node[-1]])
                    else:
                        index_candidate = np.random.choice([j for j in range(len(_open[i]))], num_nodes_per_cluster, replace=False)
                        stats = [_open[i][j]._f for j in index_candidate]
                        index_select = np.argmin(stats)
                        index_node.append(index_candidate[index_select])
                        value_node.append(stats[index_select])
            index = np.argmin(value_node)
            index_cluster = index_cluster[index]
            index_node = index_node[index]
            node = _open[index_cluster][index_node]
            _open[index_cluster] = _open[index_cluster][:index_node] + _open[index_cluster][index_node + 1:]
            open_size -= 1
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
                child_node = AStarTreeNode(node, child, node.get_g() + 1, -1, a)
                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
            if self._use_learned_heuristic:
                predicted_h, predicted_feature = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                predicted_h = predicted_h.reshape(-1)
                predicted_feature = predicted_feature.reshape(-1, 100)
            for i in range(len(children_to_be_evaluated)):
                f_cost = self.get_f_cost(children_to_be_evaluated[i].get_game_state(), children_to_be_evaluated[i].get_g(), predicted_h[i])
                children_to_be_evaluated[i].set_f_cost(f_cost)
                if children_to_be_evaluated[i].get_game_state() not in _closed:
                    distance = np.linalg.norm(cluster_centers - predicted_feature[i], axis=1)
                    cluster_ID = np.argmin(distance)
                    cluster_centers[cluster_ID] = (1 - self._c_weight) * cluster_centers[cluster_ID] + self._c_weight * predicted_feature[i]
                    children_to_be_evaluated[i].cluster_ID = cluster_ID
                    _open[cluster_ID].append(children_to_be_evaluated[i])
                    _closed.add(children_to_be_evaluated[i].get_game_state())
                    open_size += 1
            children_to_be_evaluated.clear()
            x_input_of_children_to_be_evaluated.clear()