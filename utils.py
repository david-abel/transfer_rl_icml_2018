# Python imports.
import itertools
import random

# Other imports
from RockSampleMDPClass import RockSampleMDP
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP, FourRoomMDP, ComboLockMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.mdp import MDPDistribution

def make_mdp_distr(mdp_class, is_goal_terminal, mdp_size=11, horizon=0, gamma=0.99):
    '''
    Args:
        mdp_class (str): one of {"grid", "random"}
        horizon (int)
        step_cost (float)
        gamma (float)

    Returns:
        (MDPDistribution)
    '''
    mdp_dist_dict = {}

    height, width, = mdp_size, mdp_size

    # Corridor.
    corr_width = 20
    corr_goal_magnitude = 1 #random.randint(1, 5)
    corr_goal_cols = [i for i in xrange(1, corr_goal_magnitude + 1)] + [j for j in xrange(corr_width-corr_goal_magnitude + 1, corr_width + 1)]
    corr_goal_locs  = list(itertools.product(corr_goal_cols, [1]))

    # Grid World
    tl_grid_world_rows, tl_grid_world_cols = [i for i in xrange(width - 4, width)], [j for j in xrange(height - 4, height)]
    tl_grid_goal_locs = list(itertools.product(tl_grid_world_rows, tl_grid_world_cols))
    tr_grid_world_rows, tr_grid_world_cols = [i for i in xrange(1, 4)], [j for j in xrange(height - 4, height)]
    tr_grid_goal_locs = list(itertools.product(tr_grid_world_rows, tr_grid_world_cols))
    grid_goal_locs = tl_grid_goal_locs + tr_grid_goal_locs

    # Four room.
    four_room_goal_locs = [(width, height), (width, 1), (1, height), (1, height - 2), (width - 2, height - 2)]#, (width - 2, 1)]

    # SPREAD vs. TIGHT
    spread_goal_locs = [(width, height), (width, 1), (1, height), (1, height - 2), (width - 2, height - 2), (width - 2, 1), (2,2)]
    tight_goal_locs = [(width, height), (width-1, height), (width, height-1), (width, height - 2), (width - 2, height), (width - 1, height-1), (width-2,height-2)]

    changing_entities = {"four_room":four_room_goal_locs,
                    "grid":grid_goal_locs,
                    "corridor":corr_goal_locs,
                    "spread":spread_goal_locs,
                    "tight":tight_goal_locs,
                    "chain":[0.0, 0.01, 0.1, 0.5, 1.0],
                    "rock":[0.01, 0.1, 1.0, 5.0, 10.0],
                    "combo":[[3,1,2],[3,2,1],[2,3,1],[3,3,1]]
                    }

    # MDP Probability.
    num_mdps = 10 if mdp_class not in changing_entities.keys() else len(changing_entities[mdp_class])
    if mdp_class == "octo":
        num_mdps = 12
    mdp_prob = 1.0 / num_mdps

    for i in xrange(num_mdps):

        new_mdp = {"octo":make_grid_world_from_file("octogrid.txt", num_goals=12, randomize=False, goal_num=i),
                    "corridor":GridWorldMDP(width=20, height=1, init_loc=(10, 1), goal_locs=[changing_entities["corridor"][i % len(changing_entities["corridor"])]], is_goal_terminal=is_goal_terminal, name="corridor"),
                    "grid":GridWorldMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["grid"][i % len(changing_entities["grid"])]], is_goal_terminal=is_goal_terminal),
                    "chain":ChainMDP(reset_val=changing_entities["chain"][i%len(changing_entities["chain"])]),
                    "spread":GridWorldMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["spread"][i % len(changing_entities["spread"])]], is_goal_terminal=is_goal_terminal, name="spread_grid"),
                    "tight":GridWorldMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["tight"][i % len(changing_entities["tight"])]], is_goal_terminal=is_goal_terminal, name="tight_grid"),
                    "four_room":FourRoomMDP(width=width, height=height, goal_locs=[changing_entities["four_room"][i % len(changing_entities["four_room"])]], is_goal_terminal=is_goal_terminal),
                    "rock":RockSampleMDP(rock_reward=changing_entities["rock"][i%len(changing_entities["rock"])]),
                    "combo":ComboLockMDP(combo=changing_entities["combo"][i%len(changing_entities["combo"])])
                    }[mdp_class]

        new_mdp.set_gamma(gamma)
        
        mdp_dist_dict[new_mdp] = mdp_prob

    return MDPDistribution(mdp_dist_dict, horizon=horizon)
