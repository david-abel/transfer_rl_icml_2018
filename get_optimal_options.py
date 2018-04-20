#!/usr/bin/env python

# TODO: Works only in Python3

from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration
import pymzn
import numpy as np
from subprocess import call, Popen, PIPE

def construct_singletask_data(mdp, goals, nPO=0, nSO=0):
    # TODO: Assumes deterministic transition
    # TODO: Assumes single task planning
    assert(len(goals) > 0)

    vi = ValueIteration(mdp)  # Use VI class to enumerate states
    vi.run_vi()
    vi._compute_matrix_from_trans_func()
    # q = vi.get_q_function()
    trans_matrix = vi.trans_dict
    
    # print("trans_matrix:", trans_matrix)
    N = len(trans_matrix)  # Number of states
    nGoals = len(goals)
    T = []

    print("N:", N)
    print("nGoals:", nGoals)

    state_id = dict()
    for i, u in enumerate(trans_matrix):
        state_id[u] = i + 1

    for i, u in enumerate(trans_matrix):
        T.append(set())
        # print(u, "acitons =", len(trans_matrix[u]))
        for j, a in enumerate(trans_matrix[u]):
            # print(u, a, "resulting states =", len(trans_matrix[u][a]))
            for k, v in enumerate(trans_matrix[u][a]):
                # print(u, a, v, "=", trans_matrix[u][a][v])
                if trans_matrix[u][a][v] > 0:
                    # print("FROM", i + 1, "to", k + 1)
                    # if state_id[v] not in T[i]:
                    T[i].add(state_id[v])  # Node index starts from 1 (Minizinc is 1-indexed language)
        # print("T[", i, "] =", T[i])
        # Ti_str = list(T[i]).__str__().replace('set([','{').replace('])','}')
        # T[i] = Ti_str

    # TODO: convert states to ids
    G = goals

    K = nPO
    L = nSO
    data = {'N': N, 'nGoals': nGoals, 'T': T, 'G': G, 'K': K, 'L': L}
    return data

def find_point_options(mdp, goals, nPO):
    # Build a minizinc model
    goals_ = [x+1 for x in goals]
    zinc_data = construct_singletask_data(mdp, goals_, nPO, 0)
    print("Input model =", zinc_data)

    dzn = pymzn.dict2dzn(zinc_data, fout='grid.dzn')
    # Read in the file
    # print(dzn)
        
    # f = open('grid.dzn', 'w')
    # f.write(dznout)
    # f.close()
    # sol_dzn = pymzn.minizinc('options.mzn', data=zinc_data, output_mode='dzn')
    # print("sol_dzn=", sol_dzn)

    # dict_dzn = pymzn.dzn2dict(sol_dzn)
    # print("dict_dzn=", dict_dzn)

    print("Running minizinc...")
    call(["mzn-gecode", "options.mzn", "grid.dzn", "-o", "grid.ozn"])
    print("done")
    # p = Popen(["mzn-gecode", "options.mzn", "grid.dzn", "-o", "grid.ozn"], stdout=PIPE)
    # output = p.communicate()
    # print("output=", output)

    options = []
    
    # print("grid.ozn=")
    with open('grid.ozn', 'r') as ozn:
        LN = 0
        for line in ozn:
            if LN >= nPO:
                break
            option = []
            for word in line.split():
                option.append(word)
                # print(word)
            options.append(option)
            LN += 1

    return options
            
def main():
    mdp = GridWorldMDP(width=3, height=3, goal_locs=[], slip_prob=0.0)  # goal_locs needs to be an empty list for our purpose.

    nPO = 2
    options = find_point_options(mdp, [8,9], nPO)
    for o in options:
        print(o[0], " -> ", o[1], ", d = ", o[2])
    
            
if __name__ == "__main__":
    main()
