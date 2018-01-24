#!/usr/bin/env python

# Python imports.
from collections import defaultdict
import numpy as np
import sys
import copy

# Other imports.
# import OptimalBeliefAgentClass
from AvgValueIterationClass import AvgValueIteration
from simple_rl.utils import make_mdp
from simple_rl.mdp import MDP, MDPDistribution
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent, DelayedQLearnerAgent
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

#     print "avg_rew:"
#     for s in avg_rew:
#         print (s)
#         for a in avg_rew[s]:
#             print (a, ':', avg_rew[s][a])

    
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
        q_func = vi.get_q_function()
        # print "value =", value
        for s in q_func:
            for a in q_func[s]:
                opt_q_func[s][a] = max(opt_q_func[s][a], q_func[s][a])
    return opt_q_func


# def compute_pesimistic_q_function(mdp_distr, sample_rate=5):
#     '''
#     Instead of transferring an average Q-value, we transfer the highest Q-value in MDPs so that
#     it will not under estimate the Q-value.
#     '''
#     pes_q_func = defaultdict(lambda: defaultdict(lambda: float("inf")))
#     for mdp in mdp_distr.get_mdps():
#         # prob_of_mdp = mdp_distr.get_prob_of_mdp(mdp)
# 
#         # Get a vi instance to compute state space.
#         vi = ValueIteration(mdp, delta=0.0001, max_iterations=1000, sample_rate=sample_rate)
#         iters, value = vi.run_vi()
#         q_func = vi.get_q_function()
#         # print "value =", value
#         for s in q_func:
#             for a in q_func[s]:
#                 pes_q_func[s][a] = min(pes_q_func[s][a], q_func[s][a])
#     return pes_q_func
# 
# 
# def compute_average_q_function(mdp_distr, sample_rate=5):
#     '''
#     Instead of transferring an average Q-value, we transfer the highest Q-value in MDPs so that
#     it will not under estimate the Q-value.
#     '''
#     avg_q_func = defaultdict(lambda: defaultdict(lambda: 0))
#     for mdp in mdp_distr.get_mdps():
#         prob_of_mdp = mdp_distr.get_prob_of_mdp(mdp)
# 
#         # Get a vi instance to compute state space.
#         vi = ValueIteration(mdp, delta=0.0001, max_iterations=1000, sample_rate=sample_rate)
#         iters, value = vi.run_vi()
#         q_func = vi.get_q_function()
#         for s in q_func:
#             for a in q_func[s]:
#                 avg_q_func[s][a] = avg_q_func[s][a] + prob_of_mdp * q_func[s][a]
#     return avg_q_func


def main(mdp_class="grid", eps=0.1, open_plot=True):

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
    # ValueIterationClass.py: Get Q-table from this: implement a function which returns Q table from V table.
    # QLearnerAgentClass.py: Set q-table by set_q_function.
    q_func = avg_mdp_vi.get_q_function()
    transfer_ql_agent = QLearnerAgent(actions, epsilon=eps, name="Q-trans-avg")
    transfer_ql_agent_eps = QLearnerAgent(actions, epsilon=0.3, name="transferQ-eps0.3")
    # transfer_ql_agent_sm = QLearnerAgent(actions, explore="softmax", name="transferQ-sm")
    transfer_ql_agent.set_q_function(q_func)
    transfer_ql_agent_eps.set_q_function(q_func)
    # transfer_ql_agent_sm.set_q_function(q_func)

    transfer_fixed_agent = FixedPolicyAgent(avg_mdp_vi.policy, name="transferFixed")
    
    print "done." #, iters, value
    sys.stdout.flush()

    print "Optimistic Q Transfer"
    opt_q_func = compute_optimistic_q_function(mdp_distr)
    transfer_ql_agent_optq = QLearnerAgent(actions, epsilon=eps, name="Q-trans-max")
    transfer_ql_agent_optq.set_q_function(opt_q_func)

#     print "Pesimistic Q Transfer"
#     pes_q_func = compute_pesimistic_q_function(mdp_distr)
#     transfer_ql_agent_pesq = QLearnerAgent(actions, epsilon=eps, name="transferMinQ")
#     transfer_ql_agent_pesq.set_q_function(pes_q_func)
# 
#     print "Average Q Transfer"
#     avg_q_func = compute_average_q_function(mdp_distr)
#     transfer_ql_agent_avgq = QLearnerAgent(actions, epsilon=0.1, name="transferAvgQ")
#     transfer_ql_agent_avgq.set_q_function(avg_q_func)

    #     print "Avg V..."
    #     avg_val_vi = AvgValueIteration(mdp_distr)
    #     avg_val_vi.run_vi()

    # mdp_distr_copy = copy.deepcopy(mdp_distr)

    # Agents.
    print "Making agents...",
    sys.stdout.flush()
    # opt_stoch_policy = compute_optimal_stoch_policy(mdp_distr_copy)
    # opt_stoch_policy_agent = FixedPolicyAgent(opt_stoch_policy, name="$\hat{\pi}_S^*$")
    # opt_belief_agent = OptimalBeliefAgentClass.OptimalBeliefAgent(mdp_distr, actions)
    # vi_agent = FixedPolicyAgent(avg_mdp_vi.policy, name="$\hat{\pi}_D^*$")
    # avg_v_vi_agent = FixedPolicyAgent(avg_val_vi.policy, name="$\hat{\pi}_V^*$")
    rand_agent = RandomAgent(actions, name="$\pi^u$")
    pure_ql_agent = QLearnerAgent(actions, epsilon=eps, name="Q-0")
    qmax = 1.0 * (1 - 0.99)
    # qmax = 1.0
    pure_ql_agent_opt = QLearnerAgent(actions, epsilon=eps, default_q=qmax, name="Q-vmax")
    pure_ql_agent_eps = QLearnerAgent(actions, epsilon=0.3, name="pureQ-eps0.3")
    # pure_ql_agent_sm = QLearnerAgent(actions, explore="softmax", name="pureQ-sm")
    print "done."

    trans_delayed_ql_agent = DelayedQLearnerAgent(actions, opt_q_func, name="DelayedQ-trans-max")

    pure_delayed_ql_agent = DelayedQLearnerAgent(actions, opt_q_func, name="DelayedQ-vmax")
    pure_delayed_ql_agent.set_vmax()


    pure_rmax_agent = RMaxAgent(actions, name="RMAX-vmax")
    trans_rmax_agent = RMaxAgent(actions, name="RMAX-trans_max")
    trans_rmax_agent.set_q_function(opt_q_func)

    # print "Optimal deterministic:"
    # print_policy(avg_mdp_vi.get_states(), vi_agent.policy)
    # quit()

#     print "transfer_ql_agent Q-function"
#     for x in transfer_ql_agent.q_func:
#         print (x)
#         for y in transfer_ql_agent.q_func[x]:
#             print (y, ':', transfer_ql_agent.q_func[x][y])

#     print "transfer_ql_agent_maxq Q-function"
#     for x in transfer_ql_agent_optq.q_func:
#         print (x)
#         for y in transfer_ql_agent_optq.q_func[x]:
#             print (y, ':', transfer_ql_agent_optq.q_func[x][y])

#     print "transfer_ql_agent_avgq Q-function"
#     for x in transfer_ql_agent_avgq.q_func:
#         print (x)
#         for y in transfer_ql_agent_avgq.q_func[x]:
#             print (y, ':', transfer_ql_agent_avgq.q_func[x][y])

    # agents = [transfer_ql_agent, pure_ql_agent, transfer_ql_agent_eps, pure_ql_agent_eps,
    # agents = [pure_ql_agent, pure_ql_agent_opt,
    #           transfer_ql_agent_optq, transfer_ql_agent_avgq, transfer_fixed_agent, rand_agent]
    agents = [pure_ql_agent, pure_ql_agent_opt,
              transfer_ql_agent_optq, transfer_ql_agent,
              pure_rmax_agent, trans_rmax_agent,
              pure_delayed_ql_agent, trans_delayed_ql_agent]
    # agents = [pure_ql_agent, transfer_ql_agent_avgq]
    # agents = [opt_belief_agent]

    # Run task.
    # TODO: Function for Learning on each MDP
    run_agents_multi_task(agents, mdp_distr, task_samples=50, episodes=1, steps=100, reset_at_terminal=True, is_rec_disc_reward=False, cumulative_plot=True, open_plot=open_plot)

    # num_tasks = 5
    # for t in range(num_tasks):
    #     task = mdp_distr.sample()
    #     run_agents_on_mdp(agents, task)

if __name__ == "__main__":
    # mdp_class = "chain"
    # mdp_class = "random"
    # mdp_class = "grid"
    # mdp_class = "octo"
    # mdp_class = "hall"
    # mdp_class = "corridor"
    # mdp_class = "pblocks_grid"
    # mdp_class = "four_room"
    # mdp_class = "hrooms"
    # mdp_class = "rock"
    # mdp_class = "taxi"
    eps = 0.1
    open_plot = False
    main("chain", eps=eps, open_plot=open_plot)
    main("random", eps=eps, open_plot=open_plot)
    main("grid", eps=eps, open_plot=open_plot)
    main("octo", eps=eps, open_plot=open_plot)
    main("hall", eps=eps, open_plot=open_plot)
    main("corridor", eps=eps, open_plot=open_plot)
    main("pblocks_grid", eps=eps, open_plot=open_plot)
    main("four_room", eps=eps, open_plot=open_plot)
    main("hrooms", eps=eps, open_plot=open_plot)

