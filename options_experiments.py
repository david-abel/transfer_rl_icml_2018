#!/usr/bin/env python

# Python imports.
import sys
import time

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
from get_optimal_options import find_point_options

import matplotlib.pyplot as plt

def make_point_based_options(mdp_distr, num_options=1):
    '''
    Args:
        mdp_distr (MDPDistribution)

    Returns:
        (list): Contains Option instances.
    '''

    # TODO: generate an MDP with no goal automatically.
    mdp_nogoal = GridWorldMDP(width=2, height=4, init_loc=(1, 1), goal_locs=[])
    
    # Get all goal states.
    goal_list = set([])
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
                goal_list.add(i)
    goals = list(goal_list)

    # print("goals=", goals)
    
    vi = ValueIteration(mdp_nogoal)
    state_space = vi.get_states()
    
    option_models = find_point_options(mdp_nogoal, goals, num_options)
    options = []
    for o in option_models:
        print("PO: ", o[0], "->", o[1])
        # print("o[0] =", o[0], type(o[0]))
        # print("o[1] =", o[1], type(o[1]))
        init_s = state_space[int(o[0]) - 1]
        term_s = state_space[int(o[1]) - 1]
        # print("init_s = ", init_s)
        # print("term_s = ", term_s)
        init_predicate = Predicate(func=lambda x: x == init_s)
        term_predicate = Predicate(func=lambda x: x == term_s)
        o = Option(init_predicate=init_predicate,
                    term_predicate=term_predicate,
                    policy=aa_helpers._make_mini_mdp_option_policy(mdp),
                    term_prob=0.0)
        options.append(o)
    
    return options

def planning_experiments(open_plot=True):
    '''
    Summary:
        Runs an Option planner on a simple FourRoomMDP distribution vs. regular ValueIteration.
    '''

    # Setup MDP, Agents.
    mdp1 = GridWorldMDP(width=2, height=4, init_loc=(1, 1), goal_locs=[(2, 4)])
    mdp_distr = MDPDistribution({mdp1:1.0})
    # mdp2 = GridWorldMDP(width=2, height=4, init_loc=(1, 1), goal_locs=[(2, 0)])
    # mdp_distr = MDPDistribution({mdp1:0.5, mdp2:0.5})
    # Make goal-based option agent.
    point_options = make_point_based_options(mdp_distr)
    point_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=point_options)

    opt_data = {"val":0, "iters":0, "time":0}
    regular_data = {"val":0, "iters":0, "time":0}

    for mdp in mdp_distr.get_all_mdps():
        mdp_prob = mdp_distr.get_prob_of_mdp(mdp)

        # Make VIs.
        option_vi = AbstractValueIteration(ground_mdp=mdp, action_abstr=point_aa, bootstrap=False)
        regular_vi = ValueIteration(mdp, bootstrap=False)

        # Run and time VI.
        start_time = time.clock()
        opt_iters, opt_val, opt_diff_hist = option_vi.run_vi_hist()
        opt_time = round(time.clock() - start_time, 4)

        start_time = time.clock()
        iters, val, diff_hist = regular_vi.run_vi_hist()
        regular_time = round(time.clock() - start_time, 4)


        # Add relevant data.
        opt_data["val"] += opt_val * mdp_prob
        opt_data["iters"] += opt_iters * mdp_prob
        opt_data["time"] += opt_time * mdp_prob

        regular_data["val"] += val * mdp_prob
        regular_data["iters"] += iters * mdp_prob
        regular_data["time"] += regular_time * mdp_prob

    print("Options:\n\t val :", round(opt_data["val"],3), "\n\t iters :", round(opt_data["iters"],3), "\n\t time (s) :", round(opt_data["time"],3), "\n")
    print("Regular VI:\n\t val :", round(regular_data["val"],3), "\n\t iters :", round(regular_data["iters"],3), "\n\t time (s) :", round(regular_data["time"],3))
    print("Options bellman errors =", opt_diff_hist)
    print("Actions bellman errors =", diff_hist)
    plt.plot(opt_diff_hist)
    plt.plot(diff_hist)
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
    planning_experiments(open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
