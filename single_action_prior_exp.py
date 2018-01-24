#!/usr/bin/env python

'''
single_action_prior_exp.py

For running a single experiment comparing action prior policies in multitask RL.
'''

# Python imports.
from collections import defaultdict
import numpy as np
import sys
import copy
import argparse

# Other imports.
from AvgVAgentClass import AvgVAgent
from utils import make_mdp_distr
from simple_rl.mdp import MDP, MDPDistribution
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.planning.ValueIterationClass import ValueIteration

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
    # def avg_trans_func(s,a):
    #     s_prime_index = list(np.random.multinomial(1, avg_trans_probs[s][a].values())).index(1)
    #     s_prime = avg_trans_probs[s][a].keys()[s_prime_index]
    #     s_prime.set_terminal(False)
    #     return s_prime

    avg_mdp = MDP(actions, avg_trans_func, avg_rew_func, init_state, gamma)

    return avg_mdp

def compute_optimal_stoch_policy(mdp_distr):
    '''
    Args:
        mdp_distr (defaultdict)

    Returns:
        (lambda)
    '''

    # Key: state
    # Val: dict
        # Key: action
        # Val: probability
    policy_dict = defaultdict(lambda : defaultdict(float))

    # Compute optimal policy for each MDP.
    for mdp in mdp_distr.get_all_mdps():
        # Solve the MDP and get the optimal policy.
        vi = ValueIteration(mdp, delta=0.001, max_iterations=1000)
        iters, value = vi.run_vi()
        vi_policy = vi.policy
        states = vi.get_states()

        # Compute the probability each action is optimal in each state.        
        prob_of_mdp = mdp_distr.get_prob_of_mdp(mdp)
        for s in states:
            a_star = vi_policy(s)
            policy_dict[s][a_star] += prob_of_mdp

    # Create the lambda.
    def policy_from_dict(state):
        action_id = np.random.multinomial(1, policy_dict[state].values()).tolist().index(1)
        action = policy_dict[state].keys()[action_id]

        return action

    return policy_from_dict

def make_base_a_list_from_number(number, base_a):
    '''
    Args:
        number (int): Base ten.
        base_a (int): New base to convert to.

    Returns:
        (list): Contains @number converted to @base_a.
    '''
    
    # Make a single 32 bit word.
    result = [0 for bit in range(32)]

    for i in range(len(result)-1, -1, -1):
        quotient, remainder = divmod(number,base_a**i)
        result[len(result) - i - 1] = quotient
        number = remainder

    # Remove trailing zeros before the number.
    first_non_zero_index = next((index for index, element in enumerate(result) if element > 0), None)

    return result[first_non_zero_index:]

def make_all_fixed_policies(states, actions):
    '''
    Args:
        states (list)
        actions (list)

    Returns:
        (list): Contains all deterministic policies.
    '''
    all_policies = defaultdict(list)

    # Each policy is a length |S| list containing a number.
    # That number indicates which action to take in the index-th state.

    num_states = len(states)
    num_actions = len(actions)

    all_policies = [make_base_a_list_from_number(i,num_actions) for i in range(num_states**num_actions)]

    return all_policies

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

def get_all_fixed_policy_agents(mdp):
    '''
    Args:
        mdp (MDP)

    Returns:
        (list of Agent)
    '''
    states = mdp.get_states()
    actions = mdp.get_actions()

    all_policies = make_all_fixed_policies(states, actions)
    fixed_agents = []
    for i, p in enumerate(all_policies):
        policy = make_policy_from_action_str(p, actions, states)

        next_agent = FixedPolicyAgent(policy, name="rand-fixed-policy-" + str(i))

        fixed_agents.append(next_agent)

    return fixed_agents

def print_policy(state_space, policy, sample_rate=5):
    '''
    Args:
        state_space (list)
        policy (lambda : simple_rl.State --> str)
        sample_rate (int)
    '''

    for cur_state in state_space:
        print cur_state, "\n\t",
        for j in xrange(sample_rate):
            print policy(cur_state),
        print

def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp_class", type = str, default = "chain", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-goal_terminal", type = bool, default = False, nargs = '?', help = "Whether the goal is terminal.")
    parser.add_argument("-samples", type = int, default = 100, nargs = '?', help = "Number of samples for the experiment.")
    args = parser.parse_args()

    return args.mdp_class, args.goal_terminal, args.samples


def main():
    import OptimalBeliefAgentClass

    # Setup multitask setting.
    # R ~ D : Puddle, Rock Sample
    # G ~ D : octo, four_room
    # T ~ D : grid

    mdp_class, is_goal_terminal, samples = parse_args()

    mdp_distr = make_mdp_distr(mdp_class=mdp_class, is_goal_terminal=is_goal_terminal)
    mdp_distr.set_gamma(0.9)
    actions = mdp_distr.get_actions()

    # Compute average MDP.
    print "Making and solving avg MDP...",
    sys.stdout.flush()
    avg_mdp = compute_avg_mdp(mdp_distr)
    avg_mdp_vi = ValueIteration(avg_mdp, delta=0.001, max_iterations=1000, sample_rate=5)
    iters, value = avg_mdp_vi.run_vi()
    print "done." #, iters, value
    sys.stdout.flush()

    # Agents.
    print "Making agents...",
    sys.stdout.flush()
    mdp_distr_copy = copy.deepcopy(mdp_distr)
    opt_stoch_policy = compute_optimal_stoch_policy(mdp_distr_copy)
    opt_stoch_policy_agent = FixedPolicyAgent(opt_stoch_policy, name="$\pi_{prior}$")
    opt_belief_agent = OptimalBeliefAgentClass.OptimalBeliefAgent(mdp_distr, actions)
    vi_agent = FixedPolicyAgent(avg_mdp_vi.policy, name="$\pi_{avg}$")
    # avg_v_agent = AvgVAgent(mdp_distr)
    rand_agent = RandomAgent(actions, name="$\pi^u$")
    ql_agent = QLearnerAgent(actions)
    print "done."
    
    agents = [vi_agent, opt_stoch_policy_agent, rand_agent, opt_belief_agent]

    # Run task.
    run_agents_multi_task(agents, mdp_distr, task_samples=samples, episodes=1, steps=100, reset_at_terminal=False, is_rec_disc_reward=True, cumulative_plot=True)

if __name__ == "__main__":
    main()
