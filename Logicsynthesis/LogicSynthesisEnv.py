import numpy as np
import gym
import os
import re
import torch
import abc_py as abcPy
import torch_geometric
import torch_geometric.data
from transformers import BertTokenizer


synthesisOpToPosDic = {
     0: "refactor",
     1: "refactor -z",
     2: "rewrite",
     3: "rewrite -z",
     4: "resub",
     5: "resub -z",
     6: "balance"
}

NUM_LENGTH_EPISODES = 10
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
BERT_MODEL_NAME = './bert_base_cased'
tz = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


class LogicSynthesisEnv(gym.Env):
    def __init__(self, origAIG, libFile, logFile):
        self._abc = abcPy.AbcInterface()
        self._abc.start()
        self.orig_aig = origAIG
        self.n_actions = len(synthesisOpToPosDic.keys())
        self.name = origAIG.split('/')[-1].split('.')[0]
        if not os.path.exists('./' + self.name):
            os.mkdir(self.name)
        self.ep_length = NUM_LENGTH_EPISODES
        self.lib = libFile
        self.logFile = logFile
        self.actions = ''
        self.baselineReturn = self.getResynReturn()
        self.baselineNodes = self.get_node_num(self.orig_aig)

    def initial_state(self):
        state = self.orig_aig
        return state

    def reset(self):
        state = self.orig_aig
        return state, 0, False, None

    def take_step(self, state, depth, actions, nextState=None):
        assert int(actions[-1]) >= 0 and int(actions[-1]) < 7
        next_state = self.next_state(state, depth, actions, nextState)
        done = (depth + 1) == self.ep_length
        return next_state, 0, done, None

    def next_state(self, state, depth, actions, nextState=None):
        if nextState is None:
            nextState = './' + self.name + '/' + actions + '_' + str(depth + 1) + ".aig"
        abcRunCmd = "./yosys-abc -c \"read " + state + ";" + synthesisOpToPosDic[int(actions[-1])] + "; read_lib " + self.lib + ";  write " + nextState + "; print_stats\" > " + self.logFile
        os.system(abcRunCmd)
        return nextState

    def is_done_state(self, step_idx):
        return step_idx == self.ep_length

    def extract_init_graph(self, state):
        self._abc.read(state)
        data = {}
        numNodes = self._abc.numNodes()
        data['node_type'] = np.zeros(numNodes, dtype=int)
        data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
        edge_src_index = []
        edge_target_index = []
        for nodeIdx in range(numNodes):
            aigNode = self._abc.aigNode(nodeIdx)
            nodeType = aigNode.nodeType()
            data['num_inverted_predecessors'][nodeIdx] = 0
            if nodeType == 0 or nodeType == 2:
                data['node_type'][nodeIdx] = 0
            elif nodeType == 1:
                data['node_type'][nodeIdx] = 1
            else:
                data['node_type'][nodeIdx] = 2
                if nodeType == 4:
                    data['num_inverted_predecessors'][nodeIdx] = 1
                if nodeType == 5:
                    data['num_inverted_predecessors'][nodeIdx] = 2
            if (aigNode.hasFanin0()):
                fanin = aigNode.fanin0()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
            if (aigNode.hasFanin1()):
                fanin = aigNode.fanin1()
                edge_src_index.append(nodeIdx)
                edge_target_index.append(fanin)
        data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
        data['node_type'] = torch.tensor(data['node_type'])
        data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
        data['nodes'] = numNodes
        return data

    def extract_pytorch_graph(self, depth, actionSequence, graphData=None):
        seqSentence = str(depth) + " " + actionSequence
        data = self.createDataFromGraphAndSequence(seqSentence, graphData)
        return data

    def createDataFromGraphAndSequence(self, seqSentence, graphData=None):
        encoded = tz.encode_plus(
            text=seqSentence,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=32,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # ask the function to return PyTorch tensors
        )
        data = {}
        data['data_input'] = encoded['input_ids']
        data['data_attention'] = encoded['attention_mask']
        if not graphData == None:
            for k, v in graphData.items():
                data[k] = v
            numNodes = data['nodes']
            data = torch_geometric.data.Data.from_dict(data)
            data.num_nodes = numNodes
        else:
            data = torch_geometric.data.Data.from_dict(data)
        return data

    def getResynReturn(self):
        nextState = './' + self.name + '/' + "resyn.aig"
        nextBench = './' + self.name + '/' + "resyn.bench"
        abcRunCmd = "./yosys-abc -c \"read " + self.orig_aig + ";" + RESYN2_CMD + "read_lib " + self.lib + ";  write " + nextState + "; write_bench -l " + nextBench + "; map; topo; stime\" > " + self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
            return float(areaInformation[-9]) * float(areaInformation[-4])

    def get_return(self, state):
        abcRunCmd = "./yosys-abc -c \"read " + state + "; read_lib " + self.lib + "; map ; topo;stime \" > " + self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
            adpVal = float(areaInformation[-9]) * float(areaInformation[-4])
            return max(-1.0, (self.baselineReturn - adpVal) / self.baselineReturn)

    def get_factor_return(self, state):
        abcRunCmd = "./yosys-abc -c \"read " + state + "; read_lib " + self.lib + "; map ; topo;stime \" > " + self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
            return self.baselineReturn / (float(areaInformation[-9]) * float(areaInformation[-4]))

    def get_montecarlo_return(self, startingAction, state, depth, returnState=False, path=None):
        if startingAction is None:
            startingAction = np.random.randint(0, self.n_actions)
        synthesisCmd = synthesisOpToPosDic[startingAction] + ";"
        path += str(startingAction)
        depth += 1
        while depth + 1 <= self.ep_length:
            i = np.random.randint(0, self.n_actions)
            path += str(i)
            synthesisCmd += (synthesisOpToPosDic[i] + ';')
            depth += 1
        abcRunCmd = "./yosys-abc -c \"read " + state + "; read_lib " + self.lib + ";" + synthesisCmd + "map ; topo;stime \" > " + self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
            adpVal = float(areaInformation[-9]) * float(areaInformation[-4])
            if returnState:
                return max(-1.0, (self.baselineReturn - adpVal) / self.baselineReturn), path
            return max(-1.0, (self.baselineReturn - adpVal) / self.baselineReturn)

    def get_path_return(self, state, path):
        synthesisCmd = ''
        for i in range(len(path)):
            synthesisCmd = synthesisCmd + synthesisOpToPosDic[int(path[i])] + ';'
        abcRunCmd = "./yosys-abc -c \"read " + state + "; read_lib " + self.lib + ";" + synthesisCmd + "map ; topo;stime \" > " + self.logFile
        os.system(abcRunCmd)
        with open(self.logFile) as f:
            areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
            adpVal = float(areaInformation[-9]) * float(areaInformation[-4])
        return max(-1.0, (self.baselineReturn - adpVal) / self.baselineReturn)

    def get_node_num(self, state):
        self._abc.read(state)
        return self._abc.numNodes()
