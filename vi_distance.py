#!/usr/bin/env python

# Python imports.
import sys
import time
from collections import OrderedDict, defaultdict

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning.ValueIterationClass import ValueIteration
from ValueIterationDist import ValueIterationDist


def get_distance(mdp, epsilon=0.05):

    vi = ValueIteration(mdp)
    vi.run_vi()
    vstar = vi.value_func # dictionary of state -> float

    states = vi.get_states() # list of state

    distance = defaultdict(lambda: defaultdict(float))
    
    for s in states: # s: state
        vis = ValueIterationDist(mdp, vstar)
        vis.add_fixed_val(s, vstar[s])
        vis.run_vi()
        d_to_s = vis.distance
        for t in states:
            distance[t][s] = d_to_s[t]

    for s in states: # s: state
        for t in states: # s: state
            print "d[", s, "][", t, "]=", distance[s][t]
    return distance

def main(open_plot=True):
    mdp = GridWorldMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(5, 5)])
    D = get_distance(mdp)
    
if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
