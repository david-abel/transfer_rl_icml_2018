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
from simple_rl.abstraction import Predicate, InListPredicate, Option, PolicyFromDict
import matplotlib.pyplot as plt

def main():
    mdp = GridWorldMDP(width=2, height=5, init_loc=(1, 1), goal_locs=[(2, 4)])

    # Make VIs.
    test_vi = AbstractValueIteration(ground_mdp=mdp, bootstrap=False)
    iters, val, errors = test_vi.run_vi_hist()
    print("V = ", test_vi.value_func)
    print("#iters = ", iters)
    state_space = test_vi.get_states()
    print("#states = ", len(state_space))
    print("#actions= ", len(test_vi.actions))

    print("normal VI")
    test_vi2 = ValueIteration(mdp, bootstrap=False)
    iters, val, errors = test_vi2.run_vi_hist()
    print("V = ", test_vi2.value_func)
    print("#iters = ", iters)
    state_space = test_vi2.get_states()
    print("#states = ", len(state_space))
    print("#actions= ", len(test_vi2.actions))

if __name__ == "__main__":
    main()
