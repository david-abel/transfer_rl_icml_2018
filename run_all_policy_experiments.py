#!/usr/bin/env python

# Python imports.
import subprocess

def spawn_subproc(task, goal_terminal, samples):
	'''
	Args:
		task (str)
		goal_terminal (bool)
		samples (int)

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
	cmd = ['./single_action_prior_exp.py', \
							'-mdp_class=' + str(task), \
							'-goal_terminal=' + str(goal_terminal), \
							'-samples=' + str(samples)]

	subprocess.Popen(cmd)

def main():

	# R \sim D
	spawn_subproc(task="chain", goal_terminal=False, samples=30)
	spawn_subproc(task="lava", goal_terminal=False, samples=500)

	# G \sim D
	spawn_subproc(task="four_room", goal_terminal=True, samples=500)
	spawn_subproc(task="octo", goal_terminal=True, samples=300)

	# spawn_subproc(task="spread", goal_terminal=True, samples=100)
	# spawn_subproc(task="tight", goal_terminal=True, samples=100)

	# R, T_d \sim D
	spawn_subproc(task="walls", goal_terminal=False, samples=100)
	spawn_subproc(task="combo_lock", goal_terminal=False, samples=1000)

if __name__ == "__main__":
	main()