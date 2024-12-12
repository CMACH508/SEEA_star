# SeeA*
This is the open-source codebase the paper: __SeeA*: Efficient Exploration-Enhanced A* Search by Selective Sampling Search (NeruIPS 2024 Oral)__. The project page for this paper can be found on the <https://neurips.cc/virtual/2024/oral/97957>. Please do not hesitate to contact us if you have any problems.

## Description
In this paper, SeeA* search (short for **S**ampling-**e**xploration **e**nhanced **A***) algorithm is proposed by incorporating exploration behavior into A* search. The main contributions are summarized below.

- SeeA* search employs a selective sampling process to screen a dynamic candidate subset $\mathcal{D}$ from the set $\mathcal{O}$ of open nodes that are awaiting expansion. The next expanding node is selected from $\mathcal{D}$, and it may not be the node that has the best heuristic value in $\mathcal{O}$ and will be selected by A*, enabling SeeA* to explore other promising branches. To reduce the excessive expansion of unnecessary nodes during exploration, only the candidate node with the best heuristic value is expanded. Three sampling strategies are introduced to strike a balance between exploitation and exploration. The search efficiency is improved especially when the guiding heuristic function is not accurate enough. 

- We theoretically prove that SeeA* has superior efficiency over A* search when the heuristic value function deviates substantially from the true state value function. SeeA* achieves a reduced number of node expansions to identify the optimal path. This performance improvement becomes more pronounced as the complexity of the problems increases and the reliability of the guiding heuristics decreases.

- Experiments are conducted on two real-world applications, i.e., the retrosynthetic planning problem in organic chemistry and the logic synthesis problem in integrated circuit design, as well as the classical Sokoban game. SeeA* outperforms the state-of-the-art heuristic search algorithms in terms of the problem-solving success rate and solution quality while maintaining a low level of node expansions. 
## Retrosynthetic planning
### Preparation
The implemented of SeeA* on retrosynthetic planning task is built on the top of [Retro*](https://github.com/binghong-ml/retro_star) and [Retro*+](https://github.com/junsu-kim97/self_improved_retro)
You need to first install the Retro* lib by
```
pip install -e retro_star/packages/mlp_retrosyn
pip install -e retro_star/packages/rdchiral
pip install -e .
```
The building block molecules and cost estimator model are provided by [Retro*](https://www.dropbox.com/scl/fi/cchn0wjz8j0dqxhr0qrom/retro_data.zip?rlkey=kqz60ec7vx7087vg1o63nucyo&e=1&dl=0). The employed single-step policy model is provided by the [Retro*+](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI). Test dataset is provided in the attached file.

### Dependencies
```
rdkit==2022.9.3
torch==1.13.1
pandas==1.3.5
numpy==1.21.5
```

### Implementation
Run the SeeA algorithm under uniform sampling strategy, clustering sampling strategy, and UCT-like sampling strategy based on the following code.
```
python Uniform_sampling.py
python Cluster_sampling.py
python UCT_like_sampling.py
```

## Logic synthesis

### Preparation
Yosys package is needed to process the AIG graph from <https://github.com/YosysHQ/yosys>, which is a framework for RTL synthesis tools. The test chips are provided in the attachment. The lib file need to be downloaded from <https://drive.google.com/file/d/1asSHhyswfGoAbt2g2BEx7ZpHL_QjWE3T/view?usp=sharing>.

### Implementation
If you want to test a circuit, please run
```
python Uniform_sampling.py --chips 'alu4' --gpu 0 --thread 0
python Clustering_sampling.py --chips 'alu4' --gpu 0 --thread 0
python UCT_like_sampling.py --chips 'alu4' --gpu 0 --thread 0
```

## Sokoban
If you want to test on Sokoban, please run
```
python Uniform_sampling.py 
python Clustering_sampling.py
python UCT_like_sampling.py
```


## Lisence
- All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

- The license gives permission for academic use only.
