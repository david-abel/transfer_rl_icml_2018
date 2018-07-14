#!/usr/bin/env python

# Python imports.
from collections import defaultdict
import numpy as np
import sys
import copy
import argparse

# Other imports.
from utils import make_mdp_distr
# from simple_rl.utils import make_mdp
from simple_rl.mdp import MDP, MDPDistribution
from simple_rl.run_experiments import run_agents_lifelong, run_agents_on_mdp
from simple_rl.agents import RandomAgent, FixedPolicyAgent, DelayedQAgent
from simple_rl.planning.ValueIterationClass import ValueIteration
from agents.UpdatingRMaxAgentClass import UpdatingRMaxAgent
from agents.UpdatingDelayedQLearnerAgentClass import UpdatingDelayedQLearningAgent
from agents.UpdatingQLearnerAgentClass import UpdatingQLearnerAgent

# Rmax and Q-learning agents are slightly modified from simple_rl counterparts to implement Transfer in Lifelong RL.
from agents.RMaxAgentClass import RMaxAgent
from agents.QLearningAgentClass import QLearningAgent


def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp_class", type = str, default = "chain", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-goal_terminal", type = bool, default = False, nargs = '?', help = "Whether the goal is terminal.")
    parser.add_argument("-samples", type = int, default = 100, nargs = '?', help = "Number of samples for the experiment.")
    parser.add_argument("-agent_type", type = str, default = "q", nargs = '?', help = "Type of agents: (q, rmax, delayed-q).")
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
            
                for repeat in range(sample_rate):
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
        print(cur_state, "\n\t", end='')
        for j in xrange(sample_rate):
            print(policy(cur_state), end='')
        print("")


def get_q_func(vi):
    state_space = vi.get_states()
    q_func = defaultdict(lambda: defaultdict(lambda: 0)) # Assuming R>=0
    for s in state_space:
        for a in vi.actions:
            q_s_a = vi.get_q_value(s, a)
            q_func[s][a] = q_s_a
    return q_func


def compute_optimistic_q_function(mdp_distr, sample_rate=5):
    '''
    Instead of transferring an average Q-value, we transfer the highest Q-value in MDPs so that
    it will not under estimate the Q-value.
    '''
    opt_q_func = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for mdp in mdp_distr.get_mdps():
        # prob_of_mdp = mdp_distr.get_prob_of_mdp(mdp)
        # print("new mdp")
        # Get a vi instance to compute state space.
        vi = ValueIteration(mdp, delta=0.0001, max_iterations=1000, sample_rate=sample_rate)
        iters, value = vi.run_vi()
        q_func = get_q_func(vi)
        # print("value =", value)
        # print("qfunc =", q_func)
        for s in q_func:
            for a in q_func[s]:
                opt_q_func[s][a] = max(opt_q_func[s][a], q_func[s][a])
    return opt_q_func
    

def main(open_plot=True):
    episodes = 100
    steps = 100
    gamma = 0.95

    mdp_class, is_goal_terminal, samples, alg = parse_args()
    
    # Setup multitask setting.
    mdp_distr = make_mdp_distr(mdp_class=mdp_class, is_goal_terminal=is_goal_terminal, gamma=gamma)
    actions = mdp_distr.get_actions()

    # Compute average MDP.
    print("Making and solving avg MDP...", end='')
    sys.stdout.flush()
    avg_mdp = compute_avg_mdp(mdp_distr)
    avg_mdp_vi = ValueIteration(avg_mdp, delta=0.001, max_iterations=1000, sample_rate=5)
    iters, value = avg_mdp_vi.run_vi()

    ### Yuu

    # transfer_fixed_agent = FixedPolicyAgent(avg_mdp_vi.policy, name="transferFixed")
    rand_agent = RandomAgent(actions, name="$\\pi^u$")

    opt_q_func = compute_optimistic_q_function(mdp_distr)
    avg_q_func = get_q_func(avg_mdp_vi)

    best_v = -100  # Maximum possible value an agent can get in the environment.
    for x in opt_q_func:
        for y in opt_q_func[x]:
            best_v = max(best_v, opt_q_func[x][y])
    print("Vmax =", best_v)
    vmax = best_v
    
    vmax_func = defaultdict(lambda: defaultdict(lambda: vmax))
    

    if alg == "q":
        eps = 0.1
        lrate = 0.1
        pure_ql_agent = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="Q-0")
        pure_ql_agent_opt = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, default_q=vmax, name="Q-Vmax")
        ql_agent_upd_maxq = UpdatingQLearnerAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="Q-MaxQInit")

        transfer_ql_agent_optq = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="Q-UO")
        transfer_ql_agent_optq.set_init_q_function(opt_q_func)
        
        transfer_ql_agent_avgq = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="Q-AverageQInit")
        transfer_ql_agent_avgq.set_init_q_function(avg_q_func)

        agents = [transfer_ql_agent_optq, ql_agent_upd_maxq, transfer_ql_agent_avgq,
                  pure_ql_agent_opt, pure_ql_agent]
    elif alg == "rmax":
        """
        Note that Rmax is a model-based algorithm and is very slow compared to other model-free algorithms like Q-learning and delayed Q-learning.
        """
        known_threshold = 10
        min_experience = 5
        pure_rmax_agent = RMaxAgent(actions, gamma=gamma, horizon=known_threshold, s_a_threshold=min_experience, name="RMAX-Vmax")
        updating_trans_rmax_agent = UpdatingRMaxAgent(actions, gamma=gamma, horizon=known_threshold, s_a_threshold=min_experience, name="RMAX-MaxQInit")
        trans_rmax_agent = RMaxAgent(actions, gamma=gamma, horizon=known_threshold, s_a_threshold=min_experience, name="RMAX-UO")
        trans_rmax_agent.set_init_q_function(opt_q_func)
        agents = [trans_rmax_agent, updating_trans_rmax_agent, pure_rmax_agent, rand_agent]
    elif alg == "delayed-q":
        torelance = 0.1
        min_experience = 5
        pure_delayed_ql_agent = DelayedQAgent(actions, gamma=gamma, m=min_experience, epsilon1=torelance, name="DelayedQ-Vmax")
        pure_delayed_ql_agent.set_q_function(vmax_func)
        updating_delayed_ql_agent = UpdatingDelayedQLearningAgent(actions, default_q=vmax, gamma=gamma, m=min_experience, epsilon1=torelance, name="DelayedQ-MaxQInit")
        updating_delayed_ql_agent.set_q_function(vmax_func)
        trans_delayed_ql_agent = DelayedQAgent(actions, gamma=gamma, m=min_experience, epsilon1=torelance, name="DelayedQ-UO")
        trans_delayed_ql_agent.set_q_function(opt_q_func)
        
        agents = [pure_delayed_ql_agent, updating_delayed_ql_agent, trans_delayed_ql_agent, rand_agent]        
        # agents = [updating_delayed_ql_agent, trans_delayed_ql_agent, rand_agent]        
    elif alg == "sample-effect":
        """
        This runs a comparison of MaxQInit with different number of MDP samples to calculate the initial Q function. Note that the performance of the sampled MDP is ignored for this experiment. It reproduces the result of Figure 4 of "Policy and Value Transfer for Lifelong Reinforcement Learning".
        """
        torelance = 0.1
        min_experience = 5
        pure_delayed_ql_agent = DelayedQAgent(actions, opt_q_func, m=min_experience, epsilon1=torelance, name="DelayedQ-Vmax")
        pure_delayed_ql_agent.set_vmax()
        dql_60samples = UpdatingDelayedQLearningAgent(actions, default_q=vmax, gamma=gamma, m=min_experience, epsilon1=torelance, num_sample_tasks=60, name="$DelayedQ-MaxQInit60$")
        dql_40samples = UpdatingDelayedQLearningAgent(actions, default_q=vmax, gamma=gamma, m=min_experience, epsilon1=torelance, num_sample_tasks=40, name="$DelayedQ-MaxQInit40$")
        dql_20samples = UpdatingDelayedQLearningAgent(actions, default_q=vmax, gamma=gamma, m=min_experience, epsilon1=torelance, num_sample_tasks=20, name="$DelayedQ-MaxQInit20$")

        # Sample MDPs. Note that the performance of the sampled MDP is ignored and not included in the average in the final plot.
        run_agents_lifelong([dql_20samples], mdp_distr, samples=int(samples*1/5.0), episodes=episodes, steps=steps, reset_at_terminal=is_goal_terminal, track_disc_reward=False, cumulative_plot=True, open_plot=open_plot)
        # mdp_distr.reset_tasks()
        run_agents_lifelong([dql_40samples], mdp_distr, samples=int(samples*2/5.0), episodes=episodes, steps=steps, reset_at_terminal=is_goal_terminal, track_disc_reward=False, cumulative_plot=True, open_plot=open_plot)
        # mdp_distr.reset_tasks()
        run_agents_lifelong([dql_60samples], mdp_distr, samples=int(samples*3/5.0), episodes=episodes, steps=steps, reset_at_terminal=is_goal_terminal, track_disc_reward=False, cumulative_plot=True, open_plot=open_plot)
        # mdp_distr.reset_tasks()
        # agents = [pure_delayed_ql_agent]
        agents = [dql_60samples, dql_40samples, dql_20samples, pure_delayed_ql_agent]
    else:
        msg = "Unknown type of agent:" + alg + ". Use -agent_type (q, rmax, delayed-q)"
        assert False, msg

        
    # Run task.
    run_agents_lifelong(agents, mdp_distr, samples=samples, episodes=episodes, steps=steps, reset_at_terminal=is_goal_terminal, track_disc_reward=False, cumulative_plot=True, open_plot=open_plot)
    # run_agents_lifelong(agents, mdp_distr, samples=samples, episodes=1, steps=100, reset_at_terminal=is_goal_terminal, track_disc_reward=False, cumulative_plot=True, open_plot=open_plot)


if __name__ == "__main__":
    open_plot = True
    main(open_plot=open_plot)
