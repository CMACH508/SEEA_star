import pickle
import pandas as pd
import numpy as np
from policyNet import MLPModel
from valueNet import ValueMLP
import signal
from contextlib import contextmanager
import os
import multiprocessing
from multiprocessing import Process
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
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


class ValueFeatureMLP(nn.Module):
    def __init__(self, device):
        super(ValueFeatureMLP, self).__init__()
        model_f = './saved_model/best_epoch.pt'
        model = ValueMLP(n_layers=1, fp_dim=2048, latent_dim=128, dropout_rate=0.1, device=device).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()
        self.feature = nn.Sequential(*list(model.children())[0][:2])
        self.out = list(model.children())[0][3]

    def forward(self, fps):
        feature = self.feature(fps)
        out = self.out(feature)
        x = torch.log(1 + torch.exp(out))
        return x, torch.sigmoid(feature)


def prepare_starting_molecules():
    starting_mols = set(list(pd.read_csv('./prepare_data/origin_dict.csv')['mol']))
    return starting_mols


def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
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
    vs, features = model(fps)
    vs = vs.cpu().data.numpy()
    features = features.cpu().data.numpy()
    return vs.reshape(-1), features.reshape(len(mols), -1)


class Node:
    def __init__(self, state, feature, cluster_ID, h, prior, cost=0, action_mol=None, reaction=None, template=None, parent=None):
        self.state = state
        self.feature = feature
        self.cluster_ID = cluster_ID
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


class SearchAgent:
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device, candidate_size, num_clusters=5, weight=0.15):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.device = device
        self.weight = weight
        self.num_clusters = num_clusters
        self.cluster_centers = self.init_cluster_center()
        root_value, root_feature = value_fn(value_model, [target_mol], device)
        distance = np.linalg.norm(self.cluster_centers - root_feature[0], axis=1)
        cluster_sort_ID = np.argsort(distance)
        self.root = Node([target_mol], root_feature[0], cluster_sort_ID[0], root_value, 1.0)
        self.update_cluster(self.root)
        self.open = [[] for _ in range(num_clusters)]
        self.open_count = np.zeros(self.num_clusters)
        self.open[self.root.cluster_ID].append(self.root)
        self.open_count[self.root.cluster_ID] += 1
        self.visited_mol = {self.target_mol: (root_value[0], root_feature[0])}
        self.visited_policy = {}
        self.visited_state = {}
        self.candidate_size = candidate_size
        self.iterations, self.expanded, self.generated = 0, 0, 0

    def init_cluster_center(self):
        centers = [np.random.rand(128) for _ in range(self.num_clusters - 2)]
        centers.append(2 * np.random.randn(128))
        centers.append(2 * np.random.randn(128))
        return np.array(centers)

    def update_cluster(self, node):
        self.cluster_centers[node.cluster_ID] = (1 - self.weight) * self.cluster_centers[node.cluster_ID] + self.weight * node.feature

    def select(self):
        index_cluster, index_node, value_node = [], [], []
        num_nodes_per_cluster = int(self.candidate_size // self.num_clusters)
        for i in range(self.num_clusters):
            if len(self.open[i]) > 0:
                index_cluster.append(i)
                if len(self.open[i]) <= num_nodes_per_cluster:
                    cluster_values = [self.open[i][j].f for j in range(len(self.open[i]))]
                    index_node.append(np.argmin(cluster_values))
                    value_node.append(cluster_values[index_node[-1]])
                else:
                    index_candidate = np.random.choice([j for j in range(len(self.open[i]))], num_nodes_per_cluster, replace=False)
                    stats = [self.open[i][j].f for j in index_candidate]
                    index_select = np.argmin(stats)
                    index_node.append(index_candidate[index_select])
                    value_node.append(stats[index_select])
        index = np.argmin(value_node)
        index_cluster = index_cluster[index]
        index_node = index_node[index]
        ans = self.open[index_cluster][index_node]
        self.open[index_cluster] = self.open[index_cluster][:index_node] + self.open[index_cluster][index_node + 1:]
        return ans

    def expand(self, node):
        node.is_expanded = True
        self.visited_state['.'.join(node.state)] = node.f
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
            prior = 1 / len(expanded_policy['scores'])
            for i in range(len(expanded_policy['scores'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                reactant = reactant + node.state[1:]
                reactant = sorted(list(set(reactant)))
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol
                if len(reactant) == 0:
                    child = Node([], None, None, 0, cost=cost, prior=prior, action_mol=expanded_mol, reaction=reaction, template=template, parent=node)
                    return True, child
                else:
                    #if '.'.join(reactant) in self.visited_state:
                    #    continue
                    known_reactant = [r for r in reactant if r in self.visited_mol.keys()]
                    unknown_reactant = [r for r in reactant if r not in known_reactant]
                    if len(unknown_reactant) != 0:
                        unknown_values, unknown_features = value_fn(self.value_model, unknown_reactant, self.device)
                        for r in range(len(unknown_values)):
                            self.visited_mol[unknown_reactant[r]] = (unknown_values[r], unknown_features[r])
                    h = np.sum([self.visited_mol[mol][0] for mol in reactant])
                    feature = np.sum([self.visited_mol[mol][1] for mol in reactant], axis=0)
                    distance = np.linalg.norm(self.cluster_centers - feature, axis=1)
                    cluster_sort_ID = np.argsort(distance)
                    cluster_ID = cluster_sort_ID[0]
                    child = Node(reactant, feature, cluster_ID, h, cost=cost, prior=prior, action_mol=expanded_mol, reaction=reaction, template=template, parent=node)
                    self.update_cluster(child)
                    self.open[cluster_ID].append(child)
                    self.open_count[cluster_ID] += 1
        return False, None

    def search(self):
        while self.iterations < 500:
            expand_node = self.select()
            succ, node = self.expand(expand_node)
            if succ:
                print('Success!')
                break
        print(self.open_count)
        ans = self.num_clusters - np.sum(self.open_count == 0)
        return succ, node, self.iterations, self.expanded, self.generated, ans

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


def play(dataset, mols, thread, known_mols, value_model, expand_fn, device, candidate_size, weight):
    routes, templates, successes, depths, counts, expanded_nodes, generated_nodes, play_times, zero_clusters = [], [], [], [], [], [], [], [], []
    for mol in mols:
        try:
            with time_limits(600):
                player = SearchAgent(mol, known_mols, value_model, expand_fn, device, candidate_size, weight=weight)
                start = time.time()
                success, node, count, expanded, generated, zero_cluster = player.search()
                end = time.time()
                route, template = player.vis_synthetic_path(node)
                play_time = end - start
        except:
            success, route, template, zero_cluster = False, [], [], 0
        routes.append(route)
        templates.append(template)
        successes.append(success)
        zero_clusters.append(zero_cluster)
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
        'play_times': play_times,
        'zero_clusters': zero_clusters
    }
    filename = './test/stat_retro_uniform_' + dataset + '_' + str(thread) + '.pkl'
    with open(filename, 'wb') as writer:
        pickle.dump(ans, writer, protocol=4)


def gather(dataset, candidate_size):
    result = {
        'route': [],
        'template': [],
        'success': [],
        'depth': [],
        'counts': [],
        'generated': [],
        'expanded': [],
        'play_times': [],
        'zero_clusters': []
    }
    for i in range(28):
        file = './test/stat_retro_uniform_' + dataset + '_' + str(i) + '.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for key in result.keys():
            result[key] += data[key]
        os.remove(file)
    f = open('./test/stat_Cluster_' + dataset + '.pkl', 'wb')
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
    zero_clusters = np.sum(np.array(result['zero_clusters']) == 1)
    fr = open('result_A.txt', 'a')
    fr.write(dataset + '\t'+ str(candidate_size) + '\t' + str(success) + '\t' + str(depth_all) + '\t' + str(depth) + '\t' + str(generated) + '\t' + str(expanded) + '\t' + str(play_times) + '\t' + str(zero_clusters) + '\n')
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
        value_model = ValueFeatureMLP(device)
        one_steps.append(one_step)
        value_models.append(value_model)
        devices.append(device)
    files = ['USPTO']
    for file in files:
        for size in [50]:
            for weight in [0.25]:
                fileName = './test_dataset/' + file + '.pkl'
                with open(fileName, 'rb') as f:
                    targets = pickle.load(f)
                intervals = int(len(targets) / len(gpus))
                num_plus = len(targets) - intervals * len(gpus)
                jobs = [Process(target=play, args=(file, targets[i * (intervals + 1): (i + 1) * (intervals + 1)], i, known_mols, value_models[i], one_steps[i], devices[i], size, weight)) for i in range(num_plus)]
                for i in range(num_plus, len(gpus)):
                    jobs.append(Process(target=play, args=(file, targets[i * intervals + num_plus: (i + 1) * intervals + num_plus], i, known_mols, value_models[i], one_steps[i], devices[i], size, weight)))
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                gather(file, size)





