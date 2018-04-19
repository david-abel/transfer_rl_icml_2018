include "globals.mzn";


int: maxD = 100; % an upper bound of the diameter TODO: can remove?

int: N;      % Number of nodes
int: nGoals; % Number of goals (multitask setting) % TODO: multiple goal for a single task

set of int: Nodes = 1..N;
array[Nodes] of set of 1..N: T; % State transition model by primitive actions. j in T[i] means e(i, j) exists.

set of int: Goals = 1..nGoals; % Number of goals (multitask)
array[Goals] of 1..N: G;   % goal nodes

array[Nodes, Nodes] of var 0..maxD: d_g; % d[s, g] is a path distance from node s to a g-th goal.
array[Nodes, Goals] of var 0..maxD: d; % d[s, g] is a path distance from node s to a g-th goal with option model.

array[Goals] of var 0..maxD: eccentricity; % maximum path distance for each goal
var float: avg_eccentricity;  % Sum of diameters (multitask)

array[Goals] of var float: avg_distance;
var float: avg_avg_distance;  % Sum of diameters (multitask)

/*
  Functions
*/
% getChildDists: Given a node (from), return an array of distances of its child nodes to a goal.
% p: Point option
% s: Subgoal option
function array[int] of var int: getChildDists(array[int, int] of var int: dist, array[int] of set of int: t, array[int,int] of var int: p, var set of int: s, int: parent, int: goal) =
  array1d([if n in s 
              then dist[n, goal] % If a node n is a terminal for a subgoal option, then n is your child node.
            elseif n in t[parent]
              then dist[n, goal] % If a node n can be reached by your primitive actions
            elseif p[parent, n] == 1
              then dist[n, goal] % If there is a PointOption from parent to n. 
            else 
              maxD % You cannot reach to the node. We define a distance to unreachable nodes to maxD for performance issue.
            endif | n in 1..N]);

function array[int] of var int: getChildDistsNoOp(array[int, int] of var int: dist, array[int] of set of int: t, int: parent, int: goal) =
  array1d([if n in t[parent]
              then dist[n, goal] % If a node n can be reached by your primitive actions
            else 
              maxD % You cannot reach to the node. We define a distance to unreachable nodes to maxD for performance issue.
            endif | n in 1..N]);


/*
  Point option.
  Number of options are k.
*/
int: K; % number of point options
set of int: po = 1..K;
array[po] of var 1..N: PO_from; % State transition model by option set
array[po] of var 1..N: PO_to; % State transition model by option set
array[po] of var int: PO_dist;
array[Nodes,Nodes] of var 0..1: PO;
% array[Nodes,Nodes] of var int: POdist; % Duration for the PO to complete

constraint
  forall (i in 1..N) (
    forall (j in 1..N) (
      if exists (k in 1..K) (
        PO_from[k] == i /\ PO_to[k] == j
      ) then
        PO[i,j] = 1
      else
        PO[i,j] = 0
      endif
    )  
  );
  
constraint
  forall (k in 1..K) (
      PO_dist[k] = d_g[PO_from[k], PO_to[k]]
 );
  
% constraint
%   forall (i in 1..N) (
%     forall (j in 1..N) (
%       if PO[i,j] == 1 then
%         POdist[i, j] = d_g[i, j]
%       else
%         POdist[i, j] = 0
%       endif
%     )  
%   );
  
%constraint sum([card(o) | o in PO]) <= K; % Constraint: the number of point options is at most K.

/*
  Subgoal option.
  Number of options are k.
*/
int: L; % number of subgoal options

var set of 1..N: SO; % Terminal states for subgoal option.
constraint card(SO) <= L; % Constraint: the number of subgoal options is at most L.

/*
  MDP model
*/
constraint
  forall (j in 1..N) (
    forall ( i in 1..N ) (
      if i == j then
        d_g[i, j] = 0
      else
        d_g[i, j] = min(getChildDistsNoOp(d_g, T, i, j)) + 1
    endif
    )
  );

% Model with options
constraint
  forall (g in 1..nGoals) (
    forall ( i in 1..N ) (
      if i == G[g] then
        d[i, g] = 0
      else
        d[i, g] = min(getChildDists(d, T, PO, SO, i, g)) + 1
      endif
    )
  );


% Diameter is a maximum path length. (for planning)
constraint 
  forall (g in 1..nGoals) (
    eccentricity[g] = max(col(d, g))
  );
  
% avg_distance is an expected path length. (for reinforcement learning?)
% constraint
%   forall (g in 1..nGoals) (
%     avg_distance[g] = sum(col(d, g)) / N
%   );

constraint avg_eccentricity = sum(eccentricity) / nGoals;
%constraint avg_avg_distance = sum(avg_distance) / nGoals;

solve minimize avg_eccentricity;

% solve minimize sum_diameter;

output [show(PO_from[k]) ++ " " ++ show(PO_to[k]) ++ " " ++ show(PO_dist[k]) ++ "\n"| k in po];
% output ["d[" ++ show(n) ++ "," ++ show(g) ++ "] = " ++ show(d[n, g]) ++ "\n" | g in Goals, n in Nodes] ++
%       ["diameter[" ++ show(g) ++ "] = " ++ show(eccentricity[g]) ++ "\n" | g in Goals] ++
%       ["exp diam = " ++ show(avg_eccentricity) ++ "\n"] ++
%       ["PO = " ++ show(PO) ++ "\n"] ++
%       ["PO = " ++ show(PO_from[k]) ++ " -> " ++ show(PO_to[k]) ++ ", d = " ++ show(PO_dist[k]) ++ "\n" | k in po] ++ 
%       ["SO = " ++ show(SO) ++ "\n"];       
              %["avg-avg distance = " ++ show(avg_avg_distance) ++ "\n"] ++
       %["average distance[" ++ show(g) ++ "] = " ++ show(avg_distance[g]) ++ "\n" | g in Goals] ++
