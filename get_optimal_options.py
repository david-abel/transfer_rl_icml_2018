#!/usr/bin/env python

# TODO: Works only in Python3

from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration
import pymzn
import numpy as np
from subprocess import call, Popen, PIPE
import networkx as nx
from collections import defaultdict

def get_term_id(mdp):
    vi = ValueIteration(mdp)  # Use VI class to enumerate states
    vi.run_vi()
    vi._compute_matrix_from_trans_func()
    state_space = vi.get_states()
    for i, s in enumerate(state_space):
        if s.is_terminal():
            return i
    print("no goals found")
    return None

def get_transition_matrix(mdp):
    vi = ValueIteration(mdp)  # Use VI class to enumerate states
    vi.run_vi()
    vi._compute_matrix_from_trans_func()
    # q = vi.get_q_function()
    trans_matrix = vi.trans_dict

    state_id = dict()
    for i, u in enumerate(trans_matrix):
        state_id[u] = i

    T = np.zeros((len(trans_matrix), len(trans_matrix)), dtype=np.int8)
    for i, u in enumerate(trans_matrix):
        # print(u, "acitons =", len(trans_matrix[u]))
        for j, a in enumerate(trans_matrix[u]):
            # print(u, a, "resulting states =", len(trans_matrix[u][a]))
            for k, v in enumerate(trans_matrix[u][a]):
                # print(u, a, v, "=", trans_matrix[u][a][v])
                if trans_matrix[u][a][v] > 0:
                    # print("FROM", i + 1, "to", k + 1)
                    # if state_id[v] not in T[i]:
                    T[i][state_id[v]] = 1  # Node index starts from 1 (Minizinc is 1-indexed language)
    return T


def get_optimal_transition_matrix(mdp):
    vi = ValueIteration(mdp)  # Use VI class to enumerate states
    vi.run_vi()
    vi._compute_matrix_from_trans_func()
    trans_matrix = vi.trans_dict

    N = len(trans_matrix)  # Number of states
    Topt = np.zeros((N, N), dtype=np.int8)

    state_id = dict()
    for i, u in enumerate(trans_matrix):
        state_id[u] = i + 1

    for i, u in enumerate(trans_matrix):
        best_qval = -float("inf")
        best_actions = []
        for j, a in enumerate(trans_matrix[u]):
            qval = vi.get_q_value(u, a)
            if qval > best_qval:
                best_qval = qval
                best_actions = [a]
            elif qval == best_qval:
                best_actions.append(a)

        # print("best_actions =", best_actions)
        for a in best_actions:
            for k, v in enumerate(trans_matrix[u][a]):
                if trans_matrix[u][a][v] > 0:
                    Topt[i][state_id[v]-1] = 1

    return Topt

def construct_singletask_data(mdp, goals, nPO=0, nSO=0):
    # TODO: Assumes deterministic transition
    # TODO: Assumes single task planning
    assert(len(goals) > 0)
    Topt = get_optimal_transition_matrix(mdp) # TODO: mdp_distr
    N = Topt.shape[0]
    nGoals = len(goals)
    # TODO: convert states to ids
    G = goals

    K = nPO
    L = nSO
    data = {'N': N, 'nGoals': nGoals, 'T': Topt, 'G': G, 'K': K, 'L': L}
    return data

def find_point_options(mdp, goals, nPO):
    # Build a minizinc model
    goals_ = [x+1 for x in goals]
    zinc_data = construct_singletask_data(mdp, goals_, nPO, 0)
    print("Input model =", zinc_data)

    dzn = pymzn.dict2dzn(zinc_data, fout='grid.dzn')
    # Read in the file
    # print(dzn)
        
    print("Running minizinc...")
    # TODO: find a good solver
    call(["mzn-g12fd", "options.mzn", "grid.dzn", "-o", "grid.ozn"])
    # call(["mzn-g12fd", "--fzn-flags", "\"--time 60\"", "options.mzn", "grid.dzn", "-o", "grid.ozn"])
    print("done")

    options = []
    
    # print("grid.ozn=")
    with open('grid.ozn', 'r') as ozn:
        LN = 0
        for line in ozn:
            if LN >= nPO:
                break
            words = line.split()
            init_set = [int(words[0])]
            term_set = [int(words[1])]
            
            options.append((init_set, term_set))
            LN += 1

    return options

def find_betweenness_options(mdp, t=0.1):
    T = get_transition_matrix(mdp)

    print("T=", T)
    G = nx.from_numpy_matrix(T)
    N = G.number_of_nodes()
    M = G.number_of_edges()
    print("nodes=", N)
    print("edges=", M)

    #########################
    ## 1. Enumerate all candidate subgoals
    #########################
    subgoal_set = []
    for s in G.nodes():
        # print("s=", s)
        csv = nx.betweenness_centrality_subset(G, sources=[s], targets=G.nodes())
        # csv = nx.betweenness_centrality(G)
        # print("csv=", csv)
        for v in csv:
            if (s is not v) and (csv[v] / (N-2) > t) and (v not in subgoal_set):
                subgoal_set.append(v)

    # for s in subgoal_set:
    #     print(s, " is subgoal")
    # n_subgoals = sum(subgoal_set)
    # print(n_subgoals, "goals in total")
    # centralities = nx.betweenness_centrality(G)
    # for n in centralities:
    #     print("centrality=", centralities[n])

    #########################
    ## 2. Generate an initiation set for each subgoal
    #########################
    initiation_sets = defaultdict(list)
    support_scores = defaultdict(float)
    
    for g in subgoal_set:
        csg = nx.betweenness_centrality_subset(G, sources=G.nodes(), targets=[g])
        score = 0
        for s in G.nodes():
            if csg[s] / (N-2) > t:
                initiation_sets[g].append(s)
                score += csg[s]
        support_scores[g] = score
                
    # for g in subgoal_set:
    #     print("init set for ", g, " = ", initiation_sets[g])

    #########################
    ## 3. Filter subgoals according to their supports
    #########################
    filtered_subgoals = []

    subgoal_graph = G.subgraph(subgoal_set)
    
    sccs = nx.connected_components(subgoal_graph) # TODO: connected components are used instead of SCCs
    # sccs = nx.strongly_connected_components(G)
    for scc in sccs:
        scores = []
        goals = []
        for n in scc:
            scores.append(support_scores[n])
            goals.append(n)
            # print("score of ", n, " = ", support_scores[n])
        # scores = [support_scores[x] for x in scc]
        best_score = max(scores)
        best_goal = goals[scores.index(best_score)]
        filtered_subgoals.append(best_goal)

    options = []
    for g in filtered_subgoals:
        init_set = initiation_sets[g]
        goal_set = []
        goal_set.append(g)
        options.append((init_set, goal_set))

    return options

def find_eigenoptions(mdp, num_options=1):
    delta = 0.001 # threshold for float point error
    
    # TODO: assume that the state-space is strongly connected.
    A = get_transition_matrix(mdp)
    for n in range(A.shape[0]):
        if A[n][n] == 1:
            A[n][n] = 0 # Prune self-loops for the analysis            
    degrees = np.sum(A, axis=0)
    T = np.diag(degrees)
    Tngsqrt = np.diag(1.0 / np.sqrt(degrees))
    print("A=", A)
    print("T=", T)
    L = T - A
    print("L=", L)
    normL = np.matmul(np.matmul(Tngsqrt, L), Tngsqrt)
    print("normL=", normL)
    eigenvals, eigenvecs = np.linalg.eigh(normL)
    print("eigenvals=", eigenvals)
    print("eigenvecs=")
    print(eigenvecs)

    eigenoptions = []

    for i in range(1, eigenvals.shape[0]):
        # 1st eigenval is not useful
        maxnode = np.argwhere(eigenvecs[:,i] >= np.amax(eigenvecs[:, i])-delta) + 1
        minnode = np.argwhere(eigenvecs[:,1] <= np.amin(eigenvecs[:, 1])+delta) + 1
        init_set = list(maxnode.flatten())
        goal_set = list(minnode.flatten())
        print("init_set=", init_set)
        print("goal_set=", goal_set)
        eigenoptions.append((init_set, goal_set))
        eigenoptions.append((goal_set, init_set))
        
    return eigenoptions[0:num_options]
    # TODO: normL * eigenvec = eigenval * eigenvec;

def main():
    np.set_printoptions(precision=2, suppress=True)
    mdp = GridWorldMDP(width=1, height=6, init_loc=(1, 1), goal_locs=[(1, 6)], slip_prob=0.0)  # goal_locs needs to be an empty list for our purpose.
    eigens = find_eigenoptions(mdp)
    for i, o in enumerate(eigens):
        print("Option ", i)
        print("init=", o[0])
        print("goal=", o[1])

    betw_options = find_betweenness_options(mdp, 0.1)
    for i, o in enumerate(betw_options):
        print("Option ", i)
        print("init=", o[0])
        print("goal=", o[1])
    # print("term-id=", term_id)
    #for i, o in enumerate(betw_options):
    #    print("Option ", i)
    #    print("init=", init_sets[o])
    #    print("goal=", o)
    nPO = 1
    term_id = get_term_id(mdp)
    opt_options = find_point_options(mdp, [term_id], nPO)
    for i, o in enumerate(opt_options):
        print("Option ", i)
        print("init=", o[0])
        print("goal=", o[1])
    
            
if __name__ == "__main__":
    main()
