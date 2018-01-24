#!/usr/bin/env python

'''
single_learning_exp.py

For running a single experiment comparing learinng w/ action prior policies in multitask RL.
'''

# Python imports.
import copy
import argparse

# Other imports.
import action_prior_exp as ape
from ShapedQAgentClass import ShapedQAgent
from ShapedRMaxAgentClass import ShapedRMaxAgent
from PruneRMaxAgentClass import PruneRMaxAgent
from utils import make_mdp_distr
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent
from simple_rl.planning.ValueIterationClass import ValueIteration

def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp_class", type = str, default = "chain", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-agent_type", type = str, default = "rmax", nargs = '?', help = "One of {rmax, dql}")
    parser.add_argument("-samples", type = int, default = 100, nargs = '?', help = "Number of samples for the experiment.")
    args = parser.parse_args()

    return args.mdp_class, args.agent_type, args.samples


def main():

    # Setup environment.
    mdp_class, agent_type, samples = parse_args()
    is_goal_terminal = False
    mdp_distr = make_mdp_distr(mdp_class=mdp_class, is_goal_terminal=is_goal_terminal)
    mdp_distr.set_gamma(0.99)
    actions = mdp_distr.get_actions()

    # Compute priors.

        # Stochastic mixture.
    mdp_distr_copy = copy.deepcopy(mdp_distr)
    opt_stoch_policy = ape.compute_optimal_stoch_policy(mdp_distr_copy)

        # Avg MDP
    avg_mdp = ape.compute_avg_mdp(mdp_distr)
    avg_mdp_vi = ValueIteration(avg_mdp, delta=0.001, max_iterations=1000, sample_rate=5)
    iters, value = avg_mdp_vi.run_vi()

    # Make agents.

        # Q Learning
    ql_agent = QLearnerAgent(actions)
    shaped_ql_agent_prior = ShapedQAgent(shaping_policy=opt_stoch_policy, actions=actions, name="Prior-QLearning")
    shaped_ql_agent_avgmdp = ShapedQAgent(shaping_policy=avg_mdp_vi.policy, actions=actions, name="AvgMDP-QLearning")

        # RMax
    rmax_agent = RMaxAgent(actions)
    shaped_rmax_agent_prior = ShapedRMaxAgent(shaping_policy=opt_stoch_policy, state_space=avg_mdp_vi.get_states(), actions=actions, name="Prior-RMax")
    shaped_rmax_agent_avgmdp = ShapedRMaxAgent(shaping_policy=avg_mdp_vi.policy, state_space=avg_mdp_vi.get_states(), actions=actions, name="AvgMDP-RMax")
    prune_rmax_agent = PruneRMaxAgent(mdp_distr=mdp_distr)

    if agent_type == "rmax":
        agents = [rmax_agent, shaped_rmax_agent_prior, shaped_rmax_agent_avgmdp, prune_rmax_agent]
    else:
        agents = [ql_agent, shaped_ql_agent_prior, shaped_ql_agent_avgmdp]

    # Run task.
    run_agents_multi_task(agents, mdp_distr, task_samples=samples, episodes=1, steps=200, is_rec_disc_reward=False, verbose=True)


if __name__ == "__main__":
    main()