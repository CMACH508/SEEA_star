import numpy as np
import math
import copy
from domain.environment import Environment


class SimpleEnv(Environment):
    def __init__(self, branch, solution_path, printp=False):
        self._branch = branch
        self._solution_path = solution_path
        self._path = []
        self._printp = printp

    def copy(self):
        return copy.deepcopy(self)

    def reset(self):
        self._path = []

    def __hash__(self):
        return hash(str(self._path))

    def __eq__(self, other):
        return self._branch == other._branch and self._solution_path == other._solution_path

    def successors(self):
        actions = list(range(self._branch))
        return actions

    def successors_parent_pruning(self, op):
        return self.successors()

    def apply_action(self, action):
        if self._printp:
            print("path = {} action = {}".format(self._path, action))
        self._path.append(action)

    def is_solution(self):
        return self._path == self._solution_path

    def get_image_representation(self):
        image = np.zeros((1, 1, 1))
        return image

    def heuristic_value(self):
        h = 0
        return h

    def print(self):
        print(self._path)