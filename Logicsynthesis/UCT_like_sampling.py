import torch
import numpy as np
from LogicSynthesisValue import LogicSynthesisValue, extract_init_graph, extract_pytorch_graph
from LogicSynthesisEnv import LogicSynthesisEnv
from transformers import BertModel
from torch_geometric.loader import DataLoader
import argparse


c_PUCT = 1.38


def prepare_value_fn(device):
    model = LogicSynthesisValue(readout_type=['mean', 'max'])
    model.to(device)
    model.load_state_dict(torch.load('./modelValuef/model6.model', map_location={'cuda:0': device}))
    BERT_MODEL_NAME = './bert_base_cased'
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    bert_model.to(device)
    bert_model.eval()
    return model, bert_model


def estimate_value(model, bert_model, aigData, device):
    aigData = aigData.to(device)
    value = model(aigData, bert_model)
    return value.detach().cpu().numpy()


class MinMaxStats(object):
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, f_value, action=None, parent=None):
        self._f = f_value
        self._action = action
        self._parent = parent
        if self._parent is None:
            self._path = ""
            self._depth = 0
        else:
            self._path = self._parent._path + str(self._action)
            self._depth = self._parent._depth + 1

    def set_value(self, f_value):
        self._f = f_value

    def is_done(self):
        return self._depth == 10

    def stat(self, max_d=None, min_max_stats_f=None):
        if max_d is not None and min_max_stats_f is not None:
            return c_PUCT * np.sqrt(max_d) / (1 + self._depth) + min_max_stats_f.normalize(self._f)


class SEEA:
    def __init__(self, chip, device, thread):
        self._chip = chip
        self._n_actions = 7
        libFile = './lib/7nm/7nm.lib'
        origAIG = './mcnc/' + chip + '.aig'
        logFile = './SEEA_log/' + chip + '_' + str(thread)  + '_SEEA.log'
        self._env = LogicSynthesisEnv(origAIG=origAIG, libFile=libFile, logFile=logFile)
        self._init_graph_data = extract_init_graph(origAIG)
        self._device = device
        self._value_model, self._bert_model = prepare_value_fn(self._device)
        self._state, _, done, _ = self._env.reset()
        root = Node(0)
        self._open = self.initial_nodes([root])
        self._max_generated = 1540
        self._min_max_stats_f = MinMaxStats()
        self._min_max_stats_f.update(root._f)

    def prepare_value(self, nodes):
        inputStates = [extract_pytorch_graph(node._depth, node._path, self._init_graph_data) for node in nodes]
        inputStates = DataLoader(inputStates, batch_size=len(inputStates))
        inputStates = next(iter(inputStates))
        value = self._value_model(inputStates.to(self._device), self._bert_model)
        value = list(value.detach().cpu().numpy())
        return value

    def initial_nodes(self, nodes):
        f_values = self.prepare_value(nodes)
        for i in range(len(nodes)):
            nodes[i].set_value(f_values[i])
        return nodes

    def search(self):
        current_generated = 0
        best_val = -1
        best_state = None
        while len(self._open) > 0 and current_generated <= self._max_generated:
            if len(self._open) <= 5:
                f_value = [node._f for node in self._open]
                selected_index = np.argmax(f_value)
            else:
                max_d = np.max([node._depth for node in self._open])
                stats = [self._open[i].stat(max_d, self._min_max_stats_f) for i in range(len(self._open))]
                index = np.argsort(stats)[-5:]
                f_value = [self._open[i]._f for i in index]
                selected_index = np.argmax(f_value)
                selected_index = index[selected_index]
            selected_node = self._open[selected_index]
            self._open = self._open[: selected_index] + self._open[selected_index + 1:]
            expanded_nodes = []
            for action in range(self._n_actions):
                node = Node(0, action=action, parent=selected_node)
                expanded_nodes.append(node)
            expanded_nodes = self.initial_nodes(expanded_nodes)
            for node in expanded_nodes:
                self._min_max_stats_f.update(node._f)
            if expanded_nodes[0].is_done():
                current_generated += 7
                rewards = [self._env.get_path_return(self._state, node._path) for node in expanded_nodes]
                best_idx = np.argmax(rewards)
                if rewards[best_idx] > best_val:
                    best_val = rewards[best_idx]
                    best_state = expanded_nodes[best_idx]._path
                fr = open('./SEEA_result/result_' + self._chip + '_SEEA.txt', 'a')
                fr.write(str(rewards[best_idx]) + '\t' + expanded_nodes[best_idx]._path + '\t' + str(current_generated) + '\n')
                fr.close()
            else:
                self._open = self._open + expanded_nodes
        return best_val, best_state, current_generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chips', required=True, nargs='+', help='Chip name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--thread', type=int, default=0)
    args = parser.parse_args()
    device = 'cuda:' + str(args.gpu)
    for chip in args.chips:
        player = SEEA(chip, device, args.thread)
        best_val, best_state, generated = player.search()
        line = chip + '\t' + str(best_val) + '\t' + str(best_state) + '\t' + str(generated) + '\n'
        fr = open('result_SEEA.txt', 'a')
        fr.write(line)
        fr.close()
        