#!/usr/bin/env python

# Python imports.
import sys
import time
from collections import OrderedDict

# Other imports.
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.tasks import FourRoomMDP
from simple_rl.tasks import GridWorldMDP
from simple_rl.mdp import MDPDistribution
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.planning.ValueIterationClass import ValueIteration
    
    # Abstraction
from simple_rl.abstraction import AbstractionWrapper, aa_helpers, ActionAbstraction, AbstractValueIteration
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PolicyFromDictClass import PolicyFromDict
from get_optimal_options import find_point_options, find_betweenness_options, find_eigenoptions

import matplotlib.pyplot as plt

def generate_options(mdp, models):
    vi = ValueIteration(mdp)
    state_space = vi.get_states()

    # Convert a list of init states and term states into Option in simple_rl.
    options = []
    for o in models:
        init_set = []
        term_set = []
        init_xys = []
        term_xys = []
        for i in o[0]:
            t = state_space[i - 1]
            init_set.append(t)
            init_xys.append((t.x, t.y))
        for i in o[1]:
            t = state_space[i - 1]
            term_set.append(t)
            term_xys.append((t.x, t.y))
        print("inits = ", init_xys)
        print("terms = ", term_xys)
        init_predicate = InListPredicate(init_set)
        term_predicate = InListPredicate(term_set)

        mdp_subgoal = GridWorldMDP(width=mdp.width, height=mdp.height, walls=mdp.walls, init_loc=mdp.init_loc, goal_locs=term_xys)
        # TODO: Option policy should be to a shortest path to terminal states
        option = Option(init_predicate=init_predicate,
                    term_predicate=term_predicate,
                    policy=aa_helpers._make_mini_mdp_option_policy(mdp_subgoal),
                    term_prob=0.0)
        options.append(option)
    return options

def make_optimal_point_options(mdp_distr, mdp_nogoal, num_options=1):
    '''
    Args:
        mdp_distr (MDPDistribution)

    Returns:
        (list): Contains Option instances.
    '''
    
    # Get all goal states.
    # goal_list = set([])
    goal_list = []
    for mdp in mdp_distr.get_all_mdps():
        vi = ValueIteration(mdp)
        # vi.run_vi()
        # vi._compute_matrix_from_trans_func()
        state_space = vi.get_states()
        # print("size =", len(state_space))
        for i, s in enumerate(state_space):
            # print("s=", s)
            if s.is_terminal():
                # print("terminal")
                # goal_list.add(i)
                goal_list.append(i)
    # goals = list(goal_list)
    goals = goal_list
    
    option_models = find_point_options(mdp_distr.get_all_mdps(), goals, num_options)
    options = generate_options(mdp_nogoal, option_models)
    return options

def make_betweenness_options(mdp_distr, mdp_nogoal, t=0.1, num_options=1):
    option_models = find_betweenness_options(mdp_nogoal, t)
    options = generate_options(mdp_nogoal, option_models)
    return options

def make_eigenoptions(mdp_distr, mdp_nogoal, num_options=1):
    option_models = find_eigenoptions(mdp_nogoal, num_options)
    options = generate_options(mdp_nogoal, option_models)
    return options

def planning_experiments(task='spread', open_plot=True):
    '''
    Summary:
        Runs an Option planner on a simple FourRoomMDP distribution vs. regular ValueIteration.
    '''

    # Setup MDP, Agents.

    width = 4
    height = 4
    init_loc = (1, 2) # TODO: option is buggy when the terminal is a terminal
    # goal_locs = [(5, 1)]  
    # goal_locs = [(2, 3)]
    # goal_locs = [(2, 3), (2, 3)]
    spread_goal_locs = [(width, height), (width, 1), (1, height), (1, 1)]
    tight_goal_locs = [(width, height), (width-1, height), (width, height-1), (width, height - 2), (width - 2, height), (width - 1, height - 1)]
    corridor_goal_locs = [(1, 1), (width, 1)]

    if task == 'spread':
        goal_locs = spread_goal_locs
    elif task == 'tight':
        goal_locs = tight_goal_locs
    elif task == 'corridor':
        goal_locs = corridor_goal_locs
    else:
        assert(False & "No goal specified")
    
    mdps = OrderedDict() # {MDP: probability}
    for g in goal_locs:
        mdp = GridWorldMDP(width=width, height=height, init_loc=init_loc, goal_locs=[g])
        mdps[mdp] = 1.0 / len(goal_locs)
        # vi = ValueIteration(mdp)
        # state_space = vi.get_states()
        # print("goal1 = ", g, ", prob = ", mdps[mdp])
        # print("|S|=", len(state_space))
    mdp_distr = MDPDistribution(mdps)
    mdp_nogoal = GridWorldMDP(width=width, height=height, init_loc=init_loc, goal_locs=[])


    num_options = [0, 1, 2]

    optimal_iters = []
    eigen_iters = []
    
    for nop in num_options:
    
        # Make goal-based option agent.
        point_options1 = make_optimal_point_options(mdp_distr, mdp_nogoal, nop)
        point_aa1 = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=point_options1)
	
        eigen_options = make_eigenoptions(mdp_distr, mdp_nogoal, nop)
        eigen_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=eigen_options)
        
        # bet_options = make_betweenness_options(mdp_distr, mdp_nogoal, 1)
        # bet_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=bet_options)
        
        po_data1 = {"val":0, "iters":0, "time":0}
        # po_data2 = {"val":0, "iters":0, "time":0}
        # bet_data = {"val":0, "iters":0, "time":0}
        eig_data = {"val":0, "iters":0, "time":0}
        # regular_data = {"val":0, "iters":0, "time":0}
        
        for mdp in mdp_distr.get_all_mdps():
            mdp_prob = mdp_distr.get_prob_of_mdp(mdp)
            
            # Make VIs.
            # test_vi = AbstractValueIteration(ground_mdp=mdp, bootstrap=False)
            # po_iters, po_val, po_diff_hist = test_vi.run_vi_hist()
            # print("po done")
            # print(test_vi.value_func)
                    
            print("Initializing AVIs...")
            po_vi1 = AbstractValueIteration(ground_mdp=mdp, action_abstr=point_aa1, amdp_sample_rate=1, bootstrap=False)
            # po_vi2 = AbstractValueIteration(ground_mdp=mdp, action_abstr=point_aa2, amdp_sample_rate=1, bootstrap=False)
            # bet_vi = AbstractValueIteration(ground_mdp=mdp, action_abstr=bet_aa, amdp_sample_rate=1, bootstrap=False)
            eig_vi = AbstractValueIteration(ground_mdp=mdp, action_abstr=eigen_aa, amdp_sample_rate=1, bootstrap=False)
            # regular_vi = ValueIteration(mdp, bootstrap=False)
            
            # Run and time VI.
            print("############################")
            
            # start_time = time.clock()
            # iters, val, diff_hist = regular_vi.run_vi_hist()
            # regular_time = round(time.clock() - start_time, 4)
            # print("prim done")
            
            # print("actions=", po_vi.actions)
            start_time = time.clock()
            po_iters1, po_val1, po_diff_hist1 = po_vi1.run_vi_hist()
            po_time1 = round(time.clock() - start_time, 4)
            print("po1 done")
            # print(po_vi.value_func)
            
            # start_time = time.clock()
            # po_iters2, po_val2, po_diff_hist2 = po_vi2.run_vi_hist()
            # po_time2 = round(time.clock() - start_time, 4)
            # print("po2 done")
            # # print(po_vi.value_func)
            
            # start_time = time.clock()
            # bet_iters, bet_val, bet_diff_hist = bet_vi.run_vi_hist()
            # bet_time = round(time.clock() - start_time, 4)
            # print("bet done")
            # # print(bet_vi.value_func)
            
            start_time = time.clock()
            eig_iters, eig_val, eig_diff_hist = eig_vi.run_vi_hist()
            eig_time = round(time.clock() - start_time, 4)
            print("eig done")
            # print(bet_vi.value_func)
            
            
            # Add relevant data.
            po_data1["val"] += po_val1 * mdp_prob
            po_data1["iters"] += po_iters1 * mdp_prob
            po_data1["time"] += po_time1 * mdp_prob
            
            # po_data2["val"] += po_val2 * mdp_prob
            # po_data2["iters"] += po_iters2 * mdp_prob
            # po_data2["time"] += po_time2 * mdp_prob
            
            # bet_data["val"] += bet_val * mdp_prob
            # bet_data["iters"] += bet_iters * mdp_prob
            # bet_data["time"] += bet_time * mdp_prob
            
            eig_data["val"] += eig_val * mdp_prob
            eig_data["iters"] += eig_iters * mdp_prob
            eig_data["time"] += eig_time * mdp_prob
            
            # regular_data["val"] += val * mdp_prob
            # regular_data["iters"] += iters * mdp_prob
            # regular_data["time"] += regular_time * mdp_prob
            
            print("#iters = ", po_iters1, " ", eig_iters)
            # print("mdp_prob = ", mdp_prob)
            
        print("Optimal Point Options:\n\t val :", round(po_data1["val"],3), "\n\t iters :", round(po_data1["iters"],3), "\n\t time (s) :", round(po_data1["time"],3), "\n")
        # print("2 Optimal Point Options:\n\t val :", round(po_data2["val"],3), "\n\t iters :", round(po_data2["iters"],3), "\n\t time (s) :", round(po_data2["time"],3), "\n")
        # print("Betweenness Options:\n\t val :", round(bet_data["val"],3), "\n\t iters :", round(bet_data["iters"],3), "\n\t time (s) :", round(bet_data["time"],3), "\n")
        print("Eigenoptions:\n\t val :", round(eig_data["val"],3), "\n\t iters :", round(eig_data["iters"],3), "\n\t time (s) :", round(eig_data["time"],3), "\n")
        # print("Regular VI:\n\t val :", round(regular_data["val"],3), "\n\t iters :", round(regular_data["iters"],3), "\n\t time (s) :", round(regular_data["time"],3))

        optimal_iters.append(po_data1["iters"])
        eigen_iters.append(eig_data["iters"])
        
        # print("1 Optimal Point Options bellman errors =", po_diff_hist1)
        # print("2 Optimal Point Options bellman errors =", po_diff_hist2) 
        # print("Betweenness Options bellman errors =", bet_diff_hist)
        # print("Eigenoptions bellman errors =", eig_diff_hist)
        # print("Actions bellman errors =", diff_hist)
    
    plt.plot(optimal_iters)
    plt.plot(eigen_iters)
    plt.show()
    
    

#def learning_experiments(open_plot=True):
#    '''
#    Summary:
#        Runs an Option agent on a simple FourRoomMDP distribution vs. regular agents.
#    '''
#    # Setup MDP, Agents.
#    mdp = FourRoomMDP(width=10, height=10, init_loc=(1, 1), goal_locs=[(10, 10)])
#    mdp_distr = MDPDistribution({mdp:1.0})
#    ql_agent = QLearningAgent(actions=mdp_distr.get_actions())
#    rand_agent = RandomAgent(actions=mdp_distr.get_actions())
#
#    # Make goal-based option agent.
#    point_options = make_point_based_options(mdp_distr)
#    point_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=point_options)
#    option_agent = AbstractionWrapper(QLearningAgent, actions=mdp_distr.get_actions(), action_abstr=point_aa)
#
#    # Run experiment and make plot.
#    run_agents_lifelong([ql_agent, rand_agent, option_agent], mdp_distr, samples=5, episodes=100, steps=150, open_plot=open_plot)

def main(open_plot=True):
    planning_experiments('tight', open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
