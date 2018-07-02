#!/usr/bin/env python

# Python imports.
import subprocess

def spawn_subproc(task, goal_terminal, samples, agent_type):
	'''
	Args:
		task (str)
		agent_type (str)
		samples (int)

	Summary:
		Spawns a child subprocess to run the experiment.
	'''
        cmd = ['./learning_exp.py', \
	       '-mdp_class=' + str(task), \
	       '-goal_terminal=' + str(goal_terminal), \
	       '-samples=' + str(samples), \
	       '-agent_type=' + str(agent_type)]
        
	subprocess.Popen(cmd)

def main():
    tasks = ["chain", "four_room"]
    agent_types = ["q"]
    
	# QL
    for task in tasks:
        for agent in agent_types:
            spawn_subproc(task=task, goal_terminal=False, samples=10, agent_type=agent)

if __name__ == "__main__":
	main()
