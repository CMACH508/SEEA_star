import time
from prepare import prepare_data
from search.uniform_seea import SeeA
from model import PVModel
import multiprocessing
from multiprocessing import Process
import numpy as np
gpus = [0] * 13 + [1] * 13 + [2] * 12 + [3] * 12


def solve_problems(states, planner, thread, model_path=None):
    total_expanded, total_generated, total_length, total_time, number_solved = 0, 0, 0, 0, 0
    start = time.time()
    input_size, actions, in_channel = 10, 4, 4
    device = 'cuda:' + str(gpus[thread])
    time_limit = 3600
    slack_time = 600
    budget = 100000
    nn_model = PVModel(input_size=input_size, actions=actions, in_channel=in_channel, model_path=model_path, device=device)
    for name, state in states.items():
        start_overall_time = time.time()
        args = (state, name, nn_model, budget, start_overall_time, time_limit, slack_time)
        result = planner.search(args)
        solution_length, expanded, generated, time_cost, puzzle_name = result
        if solution_length > 0:
            number_solved += 1
            total_expanded += expanded
            total_generated += generated
            total_length += solution_length
            total_time += time_cost
    end = time.time()
    with open('training_bootstrap.txt', 'a') as results_file:
        results_file.write(("{:d}, {:d}, {:d}, {:d}, {:f}, {:f} ".format(number_solved, total_length, total_expanded, total_generated, total_time, end - start)))
        results_file.write('\n')


def gather(size, cb):
    fr = open('training_bootstrap.txt', 'r')
    data = []
    for line in fr.readlines():
        line = line.strip().split(',')
        line = [float(item) for item in line]
        data.append(line)
    data = np.array(data)
    data = data[-50:, :]
    solved = np.sum(data[:, 0])
    total_length = np.sum(data[:, 1])
    total_expanded = np.sum(data[:, 2])
    total_generated = np.sum(data[:, 3])
    total_time = np.sum(data[:, 4])
    success_rate = solved / 1000
    avg_length = total_length / solved
    avg_expanded = total_expanded / solved
    avg_generated = total_generated / solved
    avg_time = total_time / solved
    with open('result.txt', 'a') as results_file:
        results_file.write(("{:d}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}".format(size, cb, success_rate, avg_length, avg_expanded, avg_generated, avg_time)))
        results_file.write('\n')


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    threads = 50
    use_heuristic = False
    use_learned_heuristic = True
    model_path = 'model_5.model'
    states = prepare_data('Sokoban')
    state_keys = list(states.keys())
    for size in  [100]:
        for c in [0.5, 0.7, 0.9]:
            states_thread = []
            planner_thread = []
            for i in range(threads):
                planner_thread.append(SeeA(use_heuristic=use_heuristic, use_learned_heuristic=use_learned_heuristic, candidate_size=size, cp=c))
                state_thread = {}
                for j in range(20 * i, 20 * i + 20):
                    state_thread[state_keys[j]] = states[state_keys[j]]
                states_thread.append(state_thread)
            jobs = [Process(target=solve_problems, args=(states_thread[i], planner_thread[i], i, model_path)) for i in range(threads)]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join()
            gather(size, c)
