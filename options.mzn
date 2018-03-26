include "globals.mzn";


int: maxD = 100; % an upper bound of the diameter TODO: can remove?

int: N;      % Number of nodes
int: nGoals; % Number of goals (multitask setting) % TODO: multiple goal for a single task

set of int: Nodes = 1..N;
array[Nodes] of set of 1..N: T; % State transition model by primitive actions. j in T[i] means e(i, j) exists.

set of int: Goals = 1..nGoals; % Number of goals (multitask)
array[Goals] of 1..N: G;   % goal nodes

array[Nodes, Goals] of var 0..maxD: d; % d[s, g] is a path distance from node s to a g-th goal.

array[Goals] of var 0..maxD: diameters; % maximum path distance for each goal
var float: avg_diameter;  % Sum of diameters (multitask)

array[Goals] of var float: avg_distance;
var float: avg_avg_distance;  % Sum of diameters (multitask)

/*
  Functions
*/
% getChildDists: Given a node (from), return an array of distances of its child nodes to a goal.
% p: Point option
% s: Subgoal option
function array[int] of var int: getChildDists(array[int, int] of var int: d, array[int] of set of int: t, array[int] of var set of int: p, var set of int: s, int: parent, int: goal) =
  array1d([if n in s 
              then d[n, goal] % If a node n is a terminal for a subgoal option, then n is your child node.
            elseif n in t[parent] union p[parent] 
              then d[n, goal] % If a node n can be reached by your primitive actions or point options
            else 
              maxD % You cannot reach to the node. We define a distance to unreachable nodes to maxD for performance issue.
            endif | n in 1..N]);


/*
  Point option.
  Number of options are k.
*/
int: K; % number of point options

array[Nodes] of var set of 1..N: PO; % State transition model by option set
constraint sum([card(o) | o in PO]) = K; % Constraint: the number of point options is K.

/*
  Subgoal option.
  Number of options are k.
*/
int: L; % number of subgoal options

var set of 1..N: SO; % Terminal states for subgoal option.
constraint card(SO) = L; % Constraint: the number of subgoal options is L.

/*
  MDP model
*/
constraint
  forall (g in 1..nGoals) (
    forall ( i in 1..N ) (
      if i = G[g] then
        d[i, g] = 0
      else
        % TODO: check row
        d[i, g] = min(getChildDists(d, T, PO, SO, i, g)) + 1
    endif
    )
  );


% Diameter is a maximum path length. (for planning)
constraint 
  forall (g in 1..nGoals) (
    diameters[g] = max(col(d, g))
  );
  
% avg_distance is an expected path length. (for reinforcement learning?)
constraint
  forall (g in 1..nGoals) (
    avg_distance[g] = sum(col(d, g)) / N
  );

constraint avg_diameter = sum(diameters) / nGoals;
constraint avg_avg_distance = sum(avg_distance) / nGoals;

solve minimize avg_diameter;

% solve minimize sum_diameter;


output ["d[" ++ show(n) ++ "," ++ show(g) ++ "] = " ++ show(d[n, g]) ++ "\n" | g in Goals, n in Nodes] ++
       ["diameter[" ++ show(g) ++ "] = " ++ show(diameters[g]) ++ "\n" | g in Goals] ++
       ["exp diam = " ++ show(avg_diameter) ++ "\n"] ++
       ["avg-avg distance = " ++ show(avg_avg_distance) ++ "\n"] ++
       ["average distance = [" ++ show(g) ++ "] = " ++ show(avg_distance[g]) ++ "\n" | g in Goals] ++
       ["PO = " ++ show(PO) ++ "\n"] ++
       ["SO = " ++ show(SO) ++ "\n"];       