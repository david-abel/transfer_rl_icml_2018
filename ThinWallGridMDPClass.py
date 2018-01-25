''' ThinWallGridMDPClass.py: Contains the ThinWallGridMDP class. '''

# Python imports.
import random

# Other imports.
from simple_rl.tasks import GridWorldMDP, GridWorldState

class ThinWallGridMDP(GridWorldMDP):
    '''
    A simple grid worl subclass where walls don't take up space.
    Instead, they sit between two states.
    '''

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        r = random.random()
        if self.slip_prob > r:
            # Flip dir.
            if action == "up":
                action = random.choice(["left", "right"])
            elif action == "down":
                action = random.choice(["left", "right"])
            elif action == "left":
                action = random.choice(["up", "down"])
            elif action == "right":
                action = random.choice(["up", "down"])

        if action == "up" and state.y < self.height and not self.is_wall(state.x, state.y, state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(state.x, state.y, state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(state.x, state.y, state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x, state.y, state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs and self.is_goal_terminal:
            next_state.set_terminal(True)

        return next_state

    # Modified for new wall implementation.
    def is_wall(self, x1, y1, x2, y2):
        '''
        Args:
            x1 (int)
            y1 (int)
            x2 (int)
            y2 (int)

        Returns:
            (bool): True iff there is a wall between (x1, y1) and (x2, y2),
        '''
        return [(x1, y1), (x2, y2)] in self.walls or [(x2, y2), (x1, y1)] in self.walls

    def __str__(self):
        return "maze" + "_h-" + str(self.height) + "_w-" + str(self.width)
