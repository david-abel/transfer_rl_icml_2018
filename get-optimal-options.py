#!/usr/bin/env python

# TODO: Works only in Python3

from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration
import pymzn
import numpy as np

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

#def clean_zinc_data():
#    pass

def main():
    mdp = GridWorldMDP(width=3, height=3, goal_locs=[], slip_prob=0.0)  # goal_locs needs to be an empty list for our purpose.

    # Build a minizinc model
    zinc_data = construct_singletask_data(mdp, [9], 1, 0)
    print("zinc_data =", zinc_data)

    dzn = pymzn.dict2dzn(zinc_data)
    # Read in the file
    print(dzn)
        
    # print "##############"
    # dznout = list(dzn).__str__().replace('set([','{').replace('])','}')
    # f = open('grid.dzn', 'w')
    # f.write(dznout)
    # f.close()
    # sol_dzn = pymzn.minizinc('options.mzn', data=zinc_data, output_mode='dzn')
    # print("sol_dzn=", sol_dzn)

    # dict_dzn = pymzn.dzn2dict(sol_dzn)
    # print("dict_dzn=", dict_dzn)

    sol = pymzn.minizinc('options.mzn', data=zinc_data, output_mode='dict')
    # print("sol=", sol)
    # print("Point options", sol[0]['PO'])
    option_model = np.array(sol[0]['PO'])
    # print("Subgoal options", s[0]['SO'])
    # print("option_model=", option_model)
    
    # Value Iteration.
    # action_seq, state_seq = vi.plan(mdp.get_init_state())
    # print("Plan for", mdp)
    # for i in range(len(action_seq)):
    #     print("\t", action_seq[i], state_seq[i])

if __name__ == "__main__":
    main()
