import torch
import numpy as np
from LogicSynthesisValue import LogicSynthesisValue, extract_init_graph, extract_pytorch_graph
from LogicSynthesisEnv import LogicSynthesisEnv
from transformers import BertModel
from torch_geometric.loader import DataLoader
import argparse


def prepare_value_fn(device):
    model = LogicSynthesisValue(readout_type=['mean', 'max'])
    model.to(device)
    model.load_state_dict(torch.load('./modelValuef/model0.model', map_location={'cuda:0': device}))
    BERT_MODEL_NAME = './bert_base_cased'
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    bert_model.to(device)
    bert_model.eval()
    return model, bert_model


def estimate_value(model, bert_model, aigData, device):
    aigData = aigData.to(device)
    value = model(aigData, bert_model)
    return value.detach().cpu().numpy()


class Node:
    def __init__(self, state, f_value, action=None, parent=None):
        self._state = state
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


class SEEA:
    def __init__(self, chip, device, thread):
        self._chip = chip
        self._n_actions = 7
        libFile = './lib/7nm/7nm.lib'
        origAIG = './arithmetic/mcnc/' + chip + '.aig'
        logFile = './SEEA_log/' + chip + '_' + str(thread)  + '_SEEA.log'
        self._env = LogicSynthesisEnv(origAIG=origAIG, libFile=libFile, logFile=logFile)
        self._init_graph_data = extract_init_graph(origAIG)
        self._device = device
        self._value_model, self._bert_model = prepare_value_fn(self._device)
        state, _, done, _ = self._env.reset()
        root = Node(origAIG, 0)
        self._open = self.initial_nodes([root])
        self.max_generated = 1536

    def initial_nodes(self, nodes):
        f_values = self.prepare_value(nodes)
        for i in range(len(nodes)):
            nodes[i].set_value(f_values[i])
        return nodes

    def prepare_value(self, nodes):
        inputStates = [extract_pytorch_graph(node._depth, node._path, self._init_graph_data) for node in nodes]
        inputStates = DataLoader(inputStates, batch_size=len(inputStates))
        inputStates = next(iter(inputStates))
        value = self._value_model(inputStates.to(self._device), self._bert_model)
        value = list(value.detach().cpu().numpy())
        return value

    def search(self):
        current_generated = 0
        best_val = -1
        best_state = None
        while len(self._open) > 0 and current_generated <= self.max_generated:
            print(current_generated)
            if len(self._open) <= 5:
                f_value = [node._f for node in self._open]
                selected_index = np.argmax(f_value)
            else:
                index = np.random.choice([i for i in range(len(self._open))], 5, replace=False)
                f_value = [self._open[i]._f for i in index]
                selected_index = np.argmax(f_value)
                selected_index = index[selected_index]
            selected_node = self._open[selected_index]
            self._open = self._open[: selected_index] + self._open[selected_index + 1:]
            expanded_nodes = []
            for action in range(self._n_actions):
                state, _, done, _ = self._env.take_step(selected_node._state, selected_node._depth, selected_node._path + str(action))
                node = Node(state, 0, action=action, parent=selected_node)
                expanded_nodes.append(node)
            expanded_nodes = self.initial_nodes(expanded_nodes)
            if expanded_nodes[0].is_done():
                current_generated += 7
                rewards = [self._env.get_return(node._state) for node in expanded_nodes]
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
