from domain.sliding_tile_puzzle import SlidingTilePuzzle
from domain.sokoban import Sokoban
from domain.witness import WitnessState
import glob


def prepare_data(domain, train=False):
    states = {}
    if domain == 'Sokoban':
        problem = []
        if train:
            puzzle_files = glob.glob('./problems/sokoban/train_50000/*.txt')
        else:
            puzzle_files = ['./problems/sokoban/test/000.txt']
        problem_id = 0
        for filename in puzzle_files:
            with open(filename, 'r') as file:
                all_problems = file.readlines()
            for line_in_problem in all_problems:
                if ';' in line_in_problem:
                    if len(problem) > 0:
                        puzzle = Sokoban(problem)
                        states['puzzle_' + str(problem_id)] = puzzle
                    problem = []
                    problem_id += 1
                elif '\n' != line_in_problem:
                    problem.append(line_in_problem.split('\n')[0])
            if len(problem) > 0:
                puzzle = Sokoban(problem)
                states['puzzle_' + str(problem_id)] = puzzle
    if domain == 'SlidingTile':
        if train:
            file = './problems/stp/puzzles_5x5_train/puzzles_50000'
        else:
            file = './problems/stp/puzzles_5x5_test/puzzles_1000'
        with open(file, 'r') as file:
            problems = file.readlines()
        problem_id = 1
        for i in range(len(problems)):
            puzzle = SlidingTilePuzzle(problems[i])
            states['puzzle_' + str(problem_id)] = puzzle
            problem_id += 1
    if domain == 'Witness':
        if train:
            file = './problems/witness/puzzles_4x4_50k_train/puzzles_50000'
        else:
            file = './problems/witness/puzzles_4x4_50k_test/puzzles_1000'
        with open(file, 'r') as file:
            puzzle = file.readlines()
        i = 0
        problem_id = 1
        while i < len(puzzle):
            k = i
            while k < len(puzzle) and puzzle[k] != '\n':
                k += 1
            s = WitnessState()
            s.read_state_from_string(puzzle[i: k])
            states['puzzle_' + str(problem_id)] = s
            i = k + 1
            problem_id += 1
    return states
