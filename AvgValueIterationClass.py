# Python imports.
import copy
from collections import defaultdict
import Queue

# Other imports.
from simple_rl.planning import ValueIteration

class AvgValueIteration(ValueIteration):

    def __init__(self, mdp_distr, name="avg_value_iter", delta=0.0001, max_iterations=200, sample_rate=3):
        self.mdp_distr = mdp_distr
        ValueIteration.__init__(self, mdp=mdp_distr.get_all_mdps()[0], delta=delta, max_iterations=max_iterations, sample_rate=sample_rate, name=name)

    def _compute_matrix_from_trans_func(self):
        if self.has_computed_matrix:
            self._compute_reachable_state_space()
            # We've already run this, just return.
            return

        self.trans_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
            # K: state
                # K: a
                    # K: s_prime
                    # V: prob

        for s in self.get_states():
            for a in self.actions:
                for sample in xrange(self.sample_rate):
                    for mdp in self.mdp_distr.get_all_mdps():
                        s_prime = self.transition_func(s, a)
                        self.trans_dict[s][a][s_prime] += (1.0 / self.sample_rate) * self.mdp_distr.get_prob_of_mdp(mdp)

        self.has_computed_matrix = True

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''

        if self.reachability_done:
            return

        state_queue = Queue.Queue()
        state_queue.put(self.init_state)
        self.states.add(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in xrange(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    for mdp in self.mdp_distr.get_all_mdps():
                        next_state = self.transition_func(s,a)

                        if next_state not in self.states:
                            self.states.add(next_state)
                            state_queue.put(next_state)

        self.reachability_done = True

    def get_q_value(self, s, a):
        '''
        Args:
            s (State)
            a (str): action

        Returns:
            (float): The Q estimate given the current value function @self.value_func.
        '''
        # Compute expected value.
        expected_future_val = 0
        for s_prime in self.trans_dict[s][a].keys():
            expected_future_val += self.trans_dict[s][a][s_prime] * self.value_func[s_prime]

        return self.reward_func(s,a) + self.gamma*expected_future_val

    def run_vi(self):
        '''
        Summary:
            Runs ValueIteration and fills in the self.value_func.           
        '''
        # Algorithm bookkeeping params.
        iterations = 0
        max_diff = float("inf")
        self._compute_matrix_from_trans_func()
        state_space = self.get_states()
        self.bellman_backups = 0

        vi_dict = {}
        for mdp in self.mdp_distr.get_all_mdps():
            vi = ValueIteration(mdp, sample_rate=3)
            vi.run_vi()
            vi_dict[mdp] = vi

        for s in state_space:
            self.value_func[s] = sum([vi_dict[mdp].get_value(s) * self.mdp_distr.get_prob_of_mdp(mdp) for mdp in self.mdp_distr.get_all_mdps()])

        # Main loop.
        while max_diff > self.delta and iterations < self.max_iterations:
            max_diff = 0
            for s in state_space:
                self.bellman_backups += 1
                if s.is_terminal():
                    continue

                max_q = float("-inf")
                for a in self.actions:
                    q_s_a = self.get_q_value(s, a)
                    max_q = q_s_a if q_s_a > max_q else max_q
                # Check terminating condition.
                max_diff = max(abs(self.value_func[s] - max_q), max_diff)

                # Update value.
                self.value_func[s] = max_q
            iterations += 1
            # print "iters, val:", iterations, max_diff

        value_of_init_state = self._compute_max_qval_action_pair(self.init_state)[0]
        
        self.has_planned = True

        return iterations, value_of_init_state

