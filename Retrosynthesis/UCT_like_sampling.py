import pickle
import torch
import pandas as pd
import numpy as np
from valueNet import ValueMLP
from policyNet import MLPModel
import signal
from contextlib import contextmanager
import os
import multiprocessing
from multiprocessing import Process
from rdkit import Chem
from rdkit.Chem import AllChem
import time


class TimeoutException(Exception):
    pass


@contextmanager
def time_limits(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def prepare_expand(gpu=-1):
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    one_step = MLPModel('./saved_model/retro_star_value_ours.ckpt', './saved_model/template_rules.dat', device=device)
    return one_step


def prepare_value(gpu=-1):
    if gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(gpu)
    model_f = './saved_model/best_epoch.pt'
    model = ValueMLP(n_layers=1, fp_dim=2048, latent_dim=128, dropout_rate=0.1, device=device).to(device)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()
    return model


def prepare_starting_molecules():
    starting_mols = set(list(pd.read_csv('./prepare_data/origin_dict.csv')['mol']))
    return starting_mols


def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1
    if pack:
        arr = np.packbits(arr)
    return arr


def batch_smiles_to_fp(s_list, fp_dim=2048):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)
    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim
    return fps


def value_fn(model, mols, device):
    num_mols = len(mols)
    fps = batch_smiles_to_fp(mols, fp_dim=2048).reshape(num_mols, -1)
    fps = torch.FloatTensor(fps).to(device)
    vs = model(fps).cpu().data.numpy()
    return vs.reshape(-1)


class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, reaction=None, template=None, parent=None, weight=1.0):
        self.state = state
        self.h = h
        self.prior = prior
        self.is_expanded = False
        self.template = template
        self.reaction = reaction
        self.action_mol = action_mol
        self.parent = parent
        if parent is not None:
            self.g = self.parent.g + cost
            self.depth = self.parent.depth + 1
        else:
            self.g = 0
            self.depth = 0
        self.f = self.g + self.h
        self.visited_time = 1
        self.weight = weight

    def stat(self, max_d=None, min_max_stats_f=None):
        if max_d is not None and min_max_stats_f is not None:
            return - self.weight * np.sqrt(max_d) / (1 + self.depth) + min_max_stats_f.normalize(self.f)
        else:
            return self.visited_time * self.g

    def update_time(self):
        self.visited_time += 1


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


class SearchAgent:
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device, candidate_size, eWeight):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.eWeight = eWeight
        root_value = value_fn(self.value_model, [target_mol], self.device)[0]
        self.visited_value = {self.target_mol: root_value}
        self.root = Node([target_mol], root_value, prior=1.0, weight=self.eWeight)
        self.open = [self.root]
        self.visited_policy = {}
        self.visited_state = [target_mol]
        self.candidate_size = candidate_size
        self.iterations = 0
        self.expanded = 0
        self.generated = 0
        self.min_max_stats_f = MinMaxStats()
        self.min_max_stats_f.update(self.root.f)

    def select(self):
        if len(self.open) <= self.candidate_size:
            stats = [self.open[i].f for i in range(len(self.open))]
            for node in self.open:
                node.update_time()
            index = np.argmin(stats)
        else:
            #index_candidate = np.random.choice([i for i in range(len(self.open))], self.candidate_size, replace=False)
            max_d = np.max([self.open[i].depth for i in range(len(self.open))])
            stats = [self.open[i].stat(max_d, self.min_max_stats_f) for i in range(len(self.open))]
            index_candidate = np.argsort(stats)[: self.candidate_size]
            stats = [self.open[i].f for i in index_candidate]
            for i in index_candidate:
                self.open[i].update_time()
            index = index_candidate[np.argmin(stats)]
        ans = self.open[index]
        self.open = self.open[: index] + self.open[index + 1:]
        return ans

    def expand(self, node):
        node.is_expanded = True
        expanded_mol = node.state[0]
        if expanded_mol in self.visited_policy.keys():
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.expand_fn.run(expanded_mol, topk=50)
            self.iterations += 1
            if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
                self.visited_policy[expanded_mol] = expanded_policy.copy()
            else:
                self.visited_policy[expanded_mol] = None
        if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
            self.expanded += 1
            self.generated += len(expanded_policy['scores'])
            for i in range(len(expanded_policy['scores'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                reactant = reactant + node.state[1:]
                reactant = sorted(list(set(reactant)))
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol
                prior = 1 / len(expanded_policy['scores'])
                if len(reactant) == 0:
                    child = Node([], 0, cost=cost, prior=prior, action_mol=expanded_mol, reaction=reaction, template=template, parent=node, weight=self.eWeight)
                    return True, child
                else:
                    known_reactant = [r for r in reactant if r in self.visited_value.keys()]
                    unknown_reactant = [r for r in reactant if r not in known_reactant]
                    if len(unknown_reactant) != 0:
                        unknown_values = value_fn(self.value_model, unknown_reactant, self.device)
                        for r in range(len(unknown_values)):
                            self.visited_value[unknown_reactant[r]] = unknown_values[r]
                    h = np.sum([self.visited_value[mol] for mol in reactant])
                    child = Node(reactant, h, cost=cost, prior=prior, action_mol=expanded_mol, reaction=reaction, template=template, parent=node, weight=self.eWeight)
                    self.min_max_stats_f.update(child.f)
                    self.open.append(child)
                    self.visited_state.append('.'.join(reactant))
        return False, None

    def search(self):
        while self.iterations < 500:
            expand_node = self.select()
            succ, node = self.expand(expand_node)
            if succ:
                print('Success!')
                break
        return succ, node, self.iterations, self.expanded, self.generated

    def vis_synthetic_path(self, node):
        if node is None:
            return [], []
        reaction_path = []
        template_path = []
        current = node
        while current is not None and current.parent is not None:
            reaction_path.append(current.reaction)
            template_path.append(current.template)
            current = current.parent
        return reaction_path[::-1], template_path[::-1]


def play(dataset, mols, thread, known_mols, value_model, expand_fn, device, candidate_size, eWeight):
    routes, templates, successes, depths, counts, expanded_nodes, generated_nodes, play_times = [], [], [], [], [], [], [], []
    for mol in mols:
        try:
            with time_limits(600):
                player = SearchAgent(mol, known_mols, value_model, expand_fn, device, candidate_size, eWeight=eWeight)
                start = time.time()
                success, node, count, expanded, generated = player.search()
                end = time.time()
                route, template = player.vis_synthetic_path(node)
                play_time = end - start
        except:
            success, route, template = False, [], []
        routes.append(route)
        templates.append(template)
        successes.append(success)
        if success:
            depths.append(node.depth)
            counts.append(count)
            expanded_nodes.append(expanded)
            generated_nodes.append(generated)
            play_times.append(play_time)
        else:
            depths.append(32)
            counts.append(-1)
            expanded_nodes.append(-1)
            generated_nodes.append(-1)
            play_times.append(600)
    ans = {
        'route': routes,
        'template': templates,
        'success': successes,
        'depth': depths,
        'counts': counts,
        'generated': generated_nodes,
        'expanded': expanded_nodes,
        'play_times': play_times
    }
    filename = './test/stat_retro_uniform_' + dataset + '_' + str(thread) + '.pkl'
    with open(filename, 'wb') as writer:
        pickle.dump(ans, writer, protocol=4)


def gather(dataset, candidate_size, eWeight):
    result = {
        'route': [],
        'template': [],
        'success': [],
        'depth': [],
        'counts': [],
        'generated': [],
        'expanded': [],
        'play_times': []
    }
    for i in range(28):
        file = './test/stat_retro_uniform_' + dataset + '_' + str(i) + '.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for key in result.keys():
            result[key] += data[key]
        os.remove(file)
    f = open('./test/stat_A_' + dataset + '_' + str(candidate_size) + '_' + str(eWeight) + '.pkl', 'wb')
    pickle.dump(result, f)
    f.close()
    success = np.mean(result['success'])
    depth = np.array(result['depth'])
    depth_all = np.mean(depth)
    depth = np.mean(depth[depth != 32])
    generated = np.array(result['generated'])
    generated = np.mean(generated[generated != -1])
    expanded = np.array(result['expanded'])
    expanded = np.mean(expanded[expanded != -1])
    play_times = np.mean(result['play_times'])
    fr = open('result_A.txt', 'a')
    fr.write(dataset + '\t'+ str(candidate_size) + '\t' + str(eWeight)+ '\t' + str(success) + '\t' + str(depth_all) + '\t' + str(depth) + '\t' + str(generated) + '\t' + str(expanded) + '\t' + str(play_times) + '\n')
    fr.close()


if __name__ == '__main__':
    known_mols = prepare_starting_molecules()
    multiprocessing.set_start_method('spawn')
    one_steps = []
    value_models = []
    devices = []
    gpus = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    for i in range(len(gpus)):
        one_step = prepare_expand(gpus[i])
        device = torch.device('cuda:' + str(gpus[i]))
        value_model = prepare_value(gpus[i])
        one_steps.append(one_step)
        value_models.append(value_model)
        devices.append(device)
    files = ['USPTO']
    for file in files:
        for size in [50]:
            for eWeight in [0.35]:
                fileName = './test_dataset/' + file + '.pkl'
                with open(fileName, 'rb') as f:
                    targets = pickle.load(f)
                intervals = int(len(targets) / len(gpus))
                num_plus = len(targets) - intervals * len(gpus)
                jobs = [Process(target=play, args=(file, targets[i * (intervals + 1): (i + 1) * (intervals + 1)], i, known_mols, value_models[i], one_steps[i], devices[i], size, eWeight)) for i in range(num_plus)]
                for i in range(num_plus, len(gpus)):
                    jobs.append(Process(target=play, args=(file, targets[i * intervals + num_plus: (i + 1) * intervals + num_plus], i, known_mols, value_models[i], one_steps[i], devices[i], size, eWeight)))
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                gather(file, size, eWeight)