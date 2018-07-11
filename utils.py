# Python imports.i
import itertools
import random

# Other imports
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
    corr_goal_cols = [i for i in range(1, corr_goal_magnitude + 1)] + [j for j in range(corr_width-corr_goal_magnitude + 1, corr_width + 1)]
    corr_goal_locs  = list(itertools.product(corr_goal_cols, [1]))

    # Grid World
    tl_grid_world_rows, tl_grid_world_cols = [i for i in range(width - 4, width)], [j for j in range(height - 4, height)]
    tl_grid_goal_locs = list(itertools.product(tl_grid_world_rows, tl_grid_world_cols))
    tr_grid_world_rows, tr_grid_world_cols = [i for i in range(1, 4)], [j for j in range(height - 4, height)]
    tr_grid_goal_locs = list(itertools.product(tr_grid_world_rows, tr_grid_world_cols))
    grid_goal_locs = tl_grid_goal_locs + tr_grid_goal_locs

    # Four room.
    four_room_goal_locs = [(width, height), (width, 1), (1, height), (1, height - 2), (width - 2, height - 2), (width - 2, 1)]

    # SPREAD vs. TIGHT
    spread_goal_locs = [(width, height), (width, 1), (1, height), (1, height - 2), (width - 2, height - 2), (width - 2, 1), (2,2)]
    tight_goal_locs = [(width, height), (width-1, height), (width, height-1), (width, height - 2), (width - 2, height), (width - 1, height-1), (width-2,height-2)]

    changing_entities = {"four_room":four_room_goal_locs,
                    "grid":grid_goal_locs,
                    "corridor":corr_goal_locs,
                    "spread":spread_goal_locs,
                    "tight":tight_goal_locs,
                    "chain":[0.0, 0.01, 0.1, 0.5, 1.0],
                    "combo_lock":[[3,1,2],[3,2,1],[2,3,1],[3,3,1]],
                    "walls":make_wall_permutations(mdp_size),
                    "lava":make_lava_permutations(mdp_size)
                    }

    # MDP Probability.
    num_mdps = 10 if mdp_class not in changing_entities.keys() else len(changing_entities[mdp_class])
    if mdp_class == "octo":
        num_mdps = 12
    mdp_prob = 1.0 / num_mdps

    for i in range(num_mdps):

        new_mdp = {"chain":ChainMDP(reset_val=changing_entities["chain"][i%len(changing_entities["chain"])]),
                   # "lava":GridWorldMDP(width=width, height=height, rand_init=False, step_cost=-0.001, lava_cost=0.0, lava_locs=changing_entities["lava"][i%len(changing_entities["lava"])], goal_locs=[(mdp_size-3, mdp_size-3)], is_goal_terminal=is_goal_terminal, name="lava_world", slip_prob=0.1),
                    "four_room":FourRoomMDP(width=width, height=height, goal_locs=[changing_entities["four_room"][i % len(changing_entities["four_room"])]], is_goal_terminal=is_goal_terminal),
                   # "octo":make_grid_world_from_file("octogrid.txt", num_goals=12, randomize=False, goal_num=i),
                    "corridor":GridWorldMDP(width=20, height=1, init_loc=(10, 1), goal_locs=[changing_entities["corridor"][i % len(changing_entities["corridor"])]], is_goal_terminal=is_goal_terminal, name="corridor"),
                    "combo_lock":ComboLockMDP(combo=changing_entities["combo_lock"][i%len(changing_entities["combo_lock"])]),
                    "spread":GridWorldMDP(width=width, height=height, rand_init=False, goal_locs=[changing_entities["spread"][i % len(changing_entities["spread"])]], is_goal_terminal=is_goal_terminal, name="spread_grid"),
                    "tight":GridWorldMDP(width=10, height=10, rand_init=False, goal_locs=[changing_entities["tight"][i % len(changing_entities["tight"])]], is_goal_terminal=is_goal_terminal, name="tight_grid"),
                    }[mdp_class]

        new_mdp.set_gamma(gamma)
        
        mdp_dist_dict[new_mdp] = mdp_prob

    return MDPDistribution(mdp_dist_dict, horizon=horizon)

def make_wall_permutations(mdp_size):
    wall_one = [[(3,1),(3,2)], [(3,6), (4,6)], [(4,4),(5,4)], [(6,5), (6,6)]]
    wall_two = [[(5,1),(5,2)], [(3,4), (4,4)], [(4,2),(5,2)], [(7,5), (7,6)]]
    wall_three = [[(2,1),(2,2)], [(4,7), (5,7)], [(3,3),(4,3)], [(4,5), (4,6)], [(7,5), (7,6)]]
    wall_four = [[(4,1),(4,2)], [(3,5), (4,5)], [(5,4),(5,5)], [(6,6), (7,6)]]
    wall_five = [[(3,3),(3,2)], [(4,6), (4,7)], [(7,2),(7,3)], [(6,5), (6,6)], [(3,4), (3,3)], [(2,5), (3,5)]]
    walls = [wall_one, wall_two, wall_three, wall_four, wall_five]
    return walls

def make_lava_permutations(mdp_size):
    lava_one = [(2,2), (3,3),(4,4),(6,2),(2,6)]
    lava_two = [(4,1),(1,4),(2,5),(5,2),(7,7)]
    lava_three = [(3,6), (3,6),(7,7),(9,8),(9,9)]
    lava_four = [(5,5), (1,8),(8,2),(3,3),(4,3)]
    lava_five = [(4,1), (2,1), (3,3), (4,3), (5,5), (6,6), (5,7), (4,5)]
    lava_lists = [lava_one, lava_two, lava_three, lava_four, lava_five]
    return lava_lists
