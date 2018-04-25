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
from get_optimal_options import find_point_options, find_betweenness_options

def make_optimal_point_options(mdp_distr, mdp_nogoal, num_options=1):
    '''
    Args:
        mdp_distr (MDPDistribution)

    Returns:
        (list): Contains Option instances.
    '''
    
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

    vi = ValueIteration(mdp_nogoal)
    state_space = vi.get_states()
    
    option_models = find_point_options(mdp_distr.get_all_mdps()[0], goals, num_options)
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
        # TODO: Option policy should be to a shortest path to the goal
        o = Option(init_predicate=init_predicate,
                    term_predicate=term_predicate,
                    policy=aa_helpers._make_mini_mdp_option_policy(mdp_distr.get_all_mdps()[0]),
                    term_prob=0.0)
        options.append(o)
    
    return options

def make_betweenness_options(mdp_distr, mdp_nogoal, t=0.1, num_options=1):
    terminal_states, init_sets = find_betweenness_options(mdp_nogoal, t)
    vi = ValueIteration(mdp_nogoal)
    state_space = vi.get_states()
    options = []
    
    for o in terminal_states:
        print("terminal=", state_space[o])
        init_s = []

        print("init=")
        for i in init_sets[o]:
            init_s.append(state_space[i])
            print("   ", state_space[i])
        init_predicate = Predicate(func=lambda x: x in init_s)
        term_predicate = Predicate(func=lambda x: x == state_space[o])
        # TODO: Option policy should be to a shortest path to the goal
        o = Option(init_predicate=init_predicate,
                    term_predicate=term_predicate,
                    policy=aa_helpers._make_mini_mdp_option_policy(mdp_distr.get_all_mdps()[0]),
                    term_prob=0.0)
        options.append(o)

    return options
        
def planning_experiments(open_plot=True):
    '''
    Summary:
        Runs an Option planner on a simple FourRoomMDP distribution vs. regular ValueIteration.
    '''

    mdp_nogoal = GridWorldMDP(width=2, height=5, init_loc=(1, 1), goal_locs=[])

    # Setup MDP, Agents.
    mdp1 = GridWorldMDP(width=2, height=5, init_loc=(1, 1), goal_locs=[(2, 4)])
    mdp_distr = MDPDistribution({mdp1:1.0})
    # mdp_distr = MDPDistribution({mdp1:0.5, mdp2:0.5})


    # Make goal-based option agent.
    point_options = make_optimal_point_options(mdp_distr, mdp_nogoal)
    point_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=point_options)

    bet_options = make_betweenness_options(mdp_distr, mdp_nogoal)
    bet_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=bet_options)
     
    print(len(point_options), " point options")
    print(len(bet_options), " bet options")
    
    po_data = {"val":0, "iters":0, "time":0}
    bet_data = {"val":0, "iters":0, "time":0}
    regular_data = {"val":0, "iters":0, "time":0}

    for mdp in mdp_distr.get_all_mdps():
        mdp_prob = mdp_distr.get_prob_of_mdp(mdp)

        # Make VIs.
        test_vi = AbstractValueIteration(ground_mdp=mdp, bootstrap=False)
        po_iters, po_val, po_diff_hist = test_vi.run_vi_hist()
        print("po done")
        print(test_vi.value_func)
        exit(0)
        

        po_vi = AbstractValueIteration(ground_mdp=mdp, action_abstr=point_aa, bootstrap=False)
        bet_vi = AbstractValueIteration(ground_mdp=mdp, action_abstr=bet_aa, bootstrap=False)
        regular_vi = ValueIteration(mdp, bootstrap=False)

        # Run and time VI.
        print("############################")
        # print("actions=", po_vi.actions)
        start_time = time.clock()
        po_iters, po_val, po_diff_hist = po_vi.run_vi_hist()
        po_time = round(time.clock() - start_time, 4)
        print("po done")
        print(po_vi.value_func)
        exit(0)

        start_time = time.clock()
        bet_iters, bet_val, bet_diff_hist = bet_vi.run_vi_hist()
        bet_time = round(time.clock() - start_time, 4)
        print("bet done")
        print(bet_vi.value_func)
        
        start_time = time.clock()
        iters, val, diff_hist = regular_vi.run_vi_hist()
        regular_time = round(time.clock() - start_time, 4)
        print("prim done")

        # Add relevant data.
        po_data["val"] += po_val * mdp_prob
        po_data["iters"] += po_iters * mdp_prob
        po_data["time"] += po_time * mdp_prob

        bet_data["val"] += bet_val * mdp_prob
        bet_data["iters"] += bet_iters * mdp_prob
        bet_data["time"] += bet_time * mdp_prob

        regular_data["val"] += val * mdp_prob
        regular_data["iters"] += iters * mdp_prob
        regular_data["time"] += regular_time * mdp_prob

    print("Optimal Point Options:\n\t val :", round(po_data["val"],3), "\n\t iters :", round(po_data["iters"],3), "\n\t time (s) :", round(po_data["time"],3), "\n")
    print("Betweenness Options:\n\t val :", round(bet_data["val"],3), "\n\t iters :", round(bet_data["iters"],3), "\n\t time (s) :", round(bet_data["time"],3), "\n")
    print("Regular VI:\n\t val :", round(regular_data["val"],3), "\n\t iters :", round(regular_data["iters"],3), "\n\t time (s) :", round(regular_data["time"],3))
    
    print("Optimal Point Options bellman errors =", po_diff_hist)
    print("Betweenness Options bellman errors =", bet_diff_hist)
    print("Actions bellman errors =", diff_hist)
    plt.plot(po_diff_hist)
    plt.plot(bet_diff_hist)
    plt.plot(diff_hist)
    # plt.show()
    
def learning_experiments(open_plot=True):
   '''
   Summary:
       Runs an Option agent on a simple FourRoomMDP distribution vs. regular agents.
   '''
   # Setup MDP, Agents.
   mdp = FourRoomMDP(width=10, height=10, init_loc=(1, 1), goal_locs=[(10, 10)])
   mdp_distr = MDPDistribution({mdp:1.0})
   ql_agent = QLearningAgent(actions=mdp_distr.get_actions())
   rand_agent = RandomAgent(actions=mdp_distr.get_actions())

   # Make goal-based option agent.
   point_options = make_point_based_options(mdp_distr)
   point_aa = ActionAbstraction(prim_actions=mdp_distr.get_actions(), options=point_options)
   option_agent = AbstractionWrapper(QLearningAgent, actions=mdp_distr.get_actions(), action_abstr=point_aa)

   # Run experiment and make plot.
   run_agents_lifelong([ql_agent, rand_agent, option_agent], mdp_distr, samples=5, episodes=100, steps=150, open_plot=open_plot)


def main(open_plot=True):
    planning_experiments(open_plot)
    # learning_experiments(open_plot)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
