# Python imports.
import copy

# Other imports.
from ThinWallGridMDPClass import ThinWallGridMDP
from simple_rl.planning import ValueIteration
from simple_rl.agents import Agent
from single_action_prior_exp import compute_avg_mdp

class OptimalBeliefAgent(Agent):

    def __init__(self, mdp_distr, actions, gamma=0.99, name="$\pi_b^*$"):
        self.mdp_distr = mdp_distr
        self.active_mdp_distr = copy.deepcopy(mdp_distr)
        self.update_policy()
        Agent.__init__(self, actions=actions, gamma=gamma, name=name)

    def update_policy(self):
        avg_mdp_vi = ValueIteration(compute_avg_mdp(self.active_mdp_distr), delta=0.0001, max_iterations=1000, sample_rate=5)
        avg_mdp_vi.run_vi()
        self.policy = avg_mdp_vi.policy

    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)

        Notes:
            We assume that all Reward functions are either 0 or 1.
        '''

        if not state.is_terminal() and self.prev_action != None and len(self.active_mdp_distr.get_all_mdps()) > 1:

            # Remove falsifying MDPs.
            mdps_to_remove = self._get_falsifying_mdps(reward, state)
            if len(mdps_to_remove) > 0:
                self.active_mdp_distr.remove_mdps(mdps_to_remove)
                # Update policy.
                self.update_policy()

        self.prev_state = state
        self.prev_action = self.policy(state)

        return self.prev_action

    def _get_falsifying_mdps(self, reward, state, sample_rate=50):
        '''
        Args:
            reward (float)
            state (simple_rl.State)
        '''
        falsified_mdps = []

        for mdp in self.active_mdp_distr.get_all_mdps():
            # Grab r and s'.
            mdp_reward = mdp.get_reward_func()(self.prev_state, self.prev_action)
            mdp_next_state = mdp.get_transition_func()(self.prev_state, self.prev_action)

            if isinstance(mdp, ThinWallGridMDP) and not (mdp_next_state == state):
                falsified_mdps.append(mdp)
                continue

            if reward != mdp_reward: # and self.mdp_distr.get_num_mdps() - len(falsified_mdps) > 1:
                falsified_mdps.append(mdp)

        return falsified_mdps

    def reset(self):
    	self.active_mdp_distr = copy.deepcopy(self.mdp_distr)
        self.prev_state = None
        self.prev_action = None