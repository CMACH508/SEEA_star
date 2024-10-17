# SeeA*
This is the open-source codebase the paper: SeeA*: Efficient Exploration-Enhanced A* Search by Selective Sampling Search (NeruIPS 2024 Oral). Details will be coming soon!

## Description
 
In this paper, SeeA* search (short for **S**ampling-**e**xploration **e**nhanced **A***) algorithm is proposed by incorporating exploration behavior into A* search. The main contributions are summarized below.

- SeeA* search employs a selective sampling process to screen a dynamic candidate subset $\mathcal{D}$ from the set $\mathcal{O}$ of open nodes that are awaiting expansion. The next expanding node is selected from $\mathcal{D}$, and it may not be the node that has the best heuristic value in $\mathcal{O}$ and will be selected by A*, enabling SeeA* to explore other promising branches. To reduce the excessive expansion of unnecessary nodes during exploration, only the candidate node with the best heuristic value is expanded. Three sampling strategies are introduced to strike a balance between exploitation and exploration. The search efficiency is improved especially when the guiding heuristic function is not accurate enough. 

- We theoretically prove that SeeA* has superior efficiency over A* search when the heuristic value function deviates substantially from the true state value function. SeeA* achieves a reduced number of node expansions to identify the optimal path. This performance improvement becomes more pronounced as the complexity of the problems increases and the reliability of the guiding heuristics decreases.

- Experiments are conducted on two real-world applications, i.e., the retrosynthetic planning problem in organic chemistry and the logic synthesis problem in integrated circuit design, as well as the classical Sokoban game. SeeA* outperforms the state-of-the-art heuristic search algorithms in terms of the problem-solving success rate and solution quality while maintaining a low level of node expansions. 

