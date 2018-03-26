import pymzn

# TODO: Convert a MDP to a state-space graph


# s = pymzn.minizinc('point_option.mzn', data={'N': 6, 'Goal': 5, 'M': 6, 'T': [{2, 3}, {4}, {4}, {5}, {5}, {3}], 'k': 2})
s = pymzn.minizinc('options.mzn', data={'N': 5, 'nGoals': 2, 'T': [{2, 3}, {4}, {4}, {5}, {1}], 'G': [3, 5], 'K': 0, 'L': 1})

print(s)
# print(s[0])
print("Point options", s[0]['PO'])
print("Subgoal options", s[0]['SO'])
