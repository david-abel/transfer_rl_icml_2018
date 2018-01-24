#!/usr/bin/env python

# Python imports.
import subprocess

def spawn_subproc(task, goal_terminal):
	'''
	Args:
		task (str)
		goal_terminal (bool)

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
	cmd = ['./action_prior_exp.py', \
							'-mdp_class=' + str(task), \
							'-goal_terminal=' + str(goal_terminal)]

	subprocess.Popen(cmd)

def main():

	# R \sim D
	spawn_subproc(task="chain", goal_terminal=False)
	# spawn_subproc(task="lava", goal_terminal=False)

	# G \sim D
	spawn_subproc(task="four_room", goal_terminal=True)
	spawn_subproc(task="octo", goal_terminal=True)

	# R, T_d \sim D
	# spawn_subproc(task="maze", goal_terminal=False)
	spawn_subproc(task="combo_lock", goal_terminal=False)

if __name__ == "__main__":
	main()