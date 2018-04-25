#!/usr/bin/env python

from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration

def main():
    mdp1 = GridWorldMDP(width=2, height=1, init_loc=(1, 1), goal_locs=[(2, 1)], slip_prob=0.5, gamma=0.5)

    vi = ValueIteration(mdp1)
    iters, value = vi.run_vi()
    print("value=", value)
            
if __name__ == "__main__":
    main()
