#!/usr/bin/env python

# Python imports.
import subprocess

def spawn_subproc(task, agent_type, samples):
	'''
	Args:
		task (str)
		agent_type (str)
		samples (int)

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
	cmd = ['./single_learning_exp.py', \
							'-mdp_class=' + str(task), \
							'-agent_type=' + str(agent_type), \
							'-samples=' + str(samples)]

	subprocess.Popen(cmd)

def main():

	# Tight vs. Spread grid
	spawn_subproc(task="tight", agent_type="rmax", samples=10)
	spawn_subproc(task="spread", agent_type="rmax", samples=10)

	# QL
	# spawn_subproc(task="four_room", agent_type="dql", samples=500)
	# spawn_subproc(task="octo", agent_type="dql", samples=500)

if __name__ == "__main__":
	main()
