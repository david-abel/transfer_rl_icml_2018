#!/usr/bin/env python

# Python imports.
from collections import defaultdict
import numpy as np
import sys
import copy
import argparse

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.mdp import MDP, MDPDistribution
from simple_rl.run_experiments import run_agents_lifelong, run_agents_on_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearningAgent, FixedPolicyAgent, DelayedQAgent
from simple_rl.planning.ValueIterationClass import ValueIteration
from agents.UpdatingRMaxAgentClass import UpdatingRMaxAgent
from agents.UpdatingDelayedQLearningAgentClass import UpdatingDelayedQLearningAgent

def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp_class", type = str, default = "chain", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-goal_terminal", type = bool, default = False, nargs = '?', help = "Whether the goal is terminal.")
    parser.add_argument("-samples", type = int, default = 100, nargs = '?', help = "Number of samples for the experiment.")
    parser.add_argument("-agent_type", type = str, default = False, nargs = '?', help = "Type of agents: (q, rmax, delayed-q).")
    args = parser.parse_args()

    return args.mdp_class, args.goal_terminal, args.samples, args.agent_type
    

def compute_avg_mdp(mdp_distr, sample_rate=5):
    '''
    Args:
        mdp_distr (defaultdict)

    Returns:
        (MDP)
    '''

    # Get normal components.
    init_state = mdp_distr.get_init_state()
    actions = mdp_distr.get_actions()
    gamma = mdp_distr.get_gamma()
    T = mdp_distr.get_all_mdps()[0].get_transition_func()

    # Compute avg reward.
    avg_rew = defaultdict(lambda : defaultdict(float))
    avg_trans_counts = defaultdict(lambda : defaultdict(lambda : defaultdict(float))) # Stores T_i(s,a,s') * Pr(M_i)
    for mdp in mdp_distr.get_mdps():
        prob_of_mdp = mdp_distr.get_prob_of_mdp(mdp)

        # Get a vi instance to compute state space.
        vi = ValueIteration(mdp, delta=0.0001, max_iterations=2000, sample_rate=sample_rate)
        iters, value = vi.run_vi()
        states = vi.get_states()

        for s in states:
            for a in actions:
                r = mdp.reward_func(s,a)

                avg_rew[s][a] += prob_of_mdp * r
            
                for repeat in xrange(sample_rate):
                    s_prime = mdp.transition_func(s,a)
                    avg_trans_counts[s][a][s_prime] += prob_of_mdp

    avg_trans_probs = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    for s in avg_trans_counts.keys():
        for a in actions:
            for s_prime in avg_trans_counts[s][a].keys():
                avg_trans_probs[s][a][s_prime] = avg_trans_counts[s][a][s_prime] / sum(avg_trans_counts[s][a].values())

    def avg_rew_func(s,a):
        return avg_rew[s][a]
    
    avg_trans_func = T
    avg_mdp = MDP(actions, avg_trans_func, avg_rew_func, init_state, gamma)

    return avg_mdp


def make_policy_from_action_list(action_ls, actions, states):
    '''
    Args:
        action_ls (ls): Each element is a number from [0:|actions|-1],
            indicating which action to take in that state. Each index
            corresponds to the index-th state in @states.
        actions (list)
        states (list)

    Returns:
        (lambda)
    '''

    policy_dict = defaultdict(str)

    for i, s in enumerate(states):
        try:
            a = actions[action_str[i]]
        except:
            a = actions[0]
        policy_dict[s] = a

    # Create the lambda
    def policy_from_dict(state):
        return policy_dict[state]

    policy_func = policy_from_dict

    return policy_func


def print_policy(state_space, policy, sample_rate=5):

    for cur_state in state_space:
        print cur_state, "\n\t",
        for j in xrange(sample_rate):
            print policy(cur_state),
        print


def compute_optimistic_q_function(mdp_distr, sample_rate=5):
    '''
    Instead of transferring an average Q-value, we transfer the highest Q-value in MDPs so that
    it will not under estimate the Q-value.
    '''
    opt_q_func = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for mdp in mdp_distr.get_mdps():
        # prob_of_mdp = mdp_distr.get_prob_of_mdp(mdp)

        # Get a vi instance to compute state space.
        vi = ValueIteration(mdp, delta=0.0001, max_iterations=1000, sample_rate=sample_rate)
        iters, value = vi.run_vi()
        q_func = vi.get_q_func()
        # print "value =", value
        for s in q_func:
            for a in q_func[s]:
                opt_q_func[s][a] = max(opt_q_func[s][a], q_func[s][a])
    return opt_q_func


def main(eps=0.1, open_plot=True):

    mdp_class, is_goal_terminal, samples, alg = parse_args()
    
    # Setup multitask setting.
    mdp_distr = make_mdp.make_mdp_distr(mdp_class=mdp_class)
    actions = mdp_distr.get_actions()

    # Compute average MDP.
    print "Making and solving avg MDP...",
    sys.stdout.flush()
    avg_mdp = compute_avg_mdp(mdp_distr)
    avg_mdp_vi = ValueIteration(avg_mdp, delta=0.001, max_iterations=1000, sample_rate=5)
    iters, value = avg_mdp_vi.run_vi()

    ### Yuu

    transfer_fixed_agent = FixedPolicyAgent(avg_mdp_vi.policy, name="transferFixed")
    rand_agent = RandomAgent(actions, name="$\\pi^u$")

    opt_q_func = compute_optimistic_q_function(mdp_distr)
    avg_q_func = avg_mdp_vi.get_q_function()

    if alg == "q":
        pure_ql_agent = QLearningAgent(actions, epsilon=eps, name="Q-0")
        qmax = 1.0 * (1 - 0.99)
        # qmax = 1.0
        pure_ql_agent_opt = QLearningAgent(actions, epsilon=eps, default_q=qmax, name="Q-vmax")
        transfer_ql_agent_optq = QLearningAgent(actions, epsilon=eps, name="Q-trans-max")
        transfer_ql_agent_optq.set_init_q_function(opt_q_func)
        transfer_ql_agent_avgq = QLearningAgent(actions, epsilon=eps, name="Q-trans-avg")
        transfer_ql_agent_avgq.set_init_q_function(avg_q_func)

        agents = [pure_ql_agent, pure_ql_agent_opt,
                  transfer_ql_agent_optq, transfer_ql_agent_avgq]
    elif alg == "rmax":
        pure_rmax_agent = RMaxAgent(actions, name="RMAX-vmax")
        updating_trans_rmax_agent = UpdatingRMaxAgent(actions, name="RMAX-updating_max")
        trans_rmax_agent = RMaxAgent(actions, name="RMAX-trans_max")
        trans_rmax_agent.set_init_q_function(opt_q_func)
        agents = [pure_rmax_agent, updating_trans_rmax_agent, trans_rmax_agent]
    elif alg == "delayed-q":
        pure_delayed_ql_agent = DelayedQLearningAgent(actions, opt_q_func, name="DelayedQ-vmax")
        pure_delayed_ql_agent.set_vmax()
        updating_delayed_ql_agent = UpdatingDelayedQLearningAgent(actions, name="DelayedQ-updating_max")
        trans_delayed_ql_agent = DelayedQLearningAgent(actions, opt_q_func, name="DelayedQ-trans-max")
        agents = [pure_delayed_ql_agent, updating_delayed_ql_agent, trans_delayed_ql_agent]
    else:
        print "Unknown type of agents:", alg
        print "(q, rmax, delayed-q)"
        assert(False)
        
    # Run task.
    run_agents_lifelong(agents, mdp_distr, task_samples=samples, episodes=1, steps=100, reset_at_terminal=is_goal_terminal, is_rec_disc_reward=False, cumulative_plot=True, open_plot=open_plot)

if __name__ == "__main__":
    eps = 0.1
    open_plot = False
    main(eps=eps, open_plot=open_plot)
