include "globals.mzn";

int: N;      % Number of nodes
int: nGoals; % Number of goals (multitask setting) % TODO: multiple goal for a single task

int: maxD = N; % an upper bound of the diameter TODO: can remove?

set of int: Nodes = 1..N;
set of int: Goals = 1..nGoals; % Number of goals (multitask)

array[Nodes, Nodes, Goals] of 0..1: T; % State transition model by primitive actions. j in T[i] means e(i, j) exists.
array[Goals] of 1..N: G;   % goal nodes

array[Nodes, Nodes, Goals] of var 0..1: reachable; % d[s, g] is a path distance from node s to a g-th goal.
array[Nodes, Goals] of var 0..maxD: d; % d[s, g] is a path distance from node s to a g-th goal with option model.

array[Goals] of var 0..maxD: eccentricity; % maximum path distance for each goal
var int: sum_eccentricity;  % Sum of diameters (multitask)

array[Goals] of var int: sum_distance;
var int: sum_sum_distance;  % Sum of diameters (multitask)

/*
  Functions
*/
% getChildDists: Given a node (from), return an array of distances of its child nodes to a goal.
% p: Point option
% s: Subgoal option
function array[int] of var int: getChildDists(array[int, int] of var int: dist, array[int, int, int] of var 0..1: t, array[int, int] of var 0..1: p, array[int] of var 0..1: s, int: parent, int: goal) =
  array1d([if t[parent, n, goal] == 1
              then dist[n, goal] % If a node n can be reached by your primitive actions
            elseif (p[parent, n] == 1) /\ (reachable[parent, n, goal] == 1)
              then dist[n, goal] % If there is a PointOption from parent to n. 
            elseif (s[n] == 1) /\ (reachable[parent, n, goal] == 1)
              then dist[n, goal] % If a node n is a terminal for a subgoal option, then n is your child node.
            else 
              maxD % You cannot reach to the node. We define a distance to unreachable nodes to maxD for performance issue.
            endif | n in 1..N]);

% function array[int] of var int: getChildDistsNoOp(array[int, int] of var int: dist, array[int, int] of int: t, int: parent, int: goal) =
%   array1d([if t[parent, n] == 1
%               then dist[n, goal] % If a node n can be reached by your primitive actions
%             else 
%               maxD % You cannot reach to the node. We define a distance to unreachable nodes to maxD for performance issue.
%             endif | n in 1..N]);


/*
  Point option.
  Number of options are k.
*/
int: K; % number of point options
set of int: po = 1..K;
array[po] of var 1..N: PO_from; % State transition model by option set
array[po] of var 1..N: PO_to; % State transition model by option set
% array[po] of var int: PO_dist;
array[Nodes,Nodes] of var 0..1: PO;
% array[Nodes,Nodes] of var int: POdist; % Duration for the PO to complete

% constraint PO_from[1] == 1;
% constraint PO_to[1] == 6;

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
  
% constraint
%   forall (k in 1..K) (
%     PO_from[k] != PO_to[k]
%   );
  
% predicate is_child(array[int, int] of int: t, var int: from, var int: to) =
%   t[from, to] == 1;
  
% predicate is_descendant(array[int, int] of 0..1: t, 1..N: from, 1..N: to, 1..N: path_length) =
%   path_length > 0 /\ 
%   (t[from, to] == 1 \/ exists(v in 1..N) (t[from, v] == 1 /\ is_descendant(t, v, to, path_length-1)));
  
% constraint
%   forall (k in 1..K) (
%     reachable[PO_from[k], PO_to[k], g] == 1
%   );
  
% constraint
%   forall (k in 1..K) (
%       PO_dist[k] = d_g[PO_from[k], PO_to[k]]
%  );
  
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

array[Nodes] of var 0..1: SO;
% var set of 1..N: SO; % Terminal states for subgoal option.
constraint sum(SO) <= L; % Constraint: the number of subgoal options is at most L.

/*
  MDP model
*/
constraint
  forall (j in 1..N) (
    forall ( i in 1..N ) (
      forall (g in 1..nGoals) (
        if i == j then
          reachable[i, j, g] = 1
        else
          reachable[i, j, g] = ((T[i, j, g] == 1) \/ exists(v in 1..N) ((T[i, v, g] == 1) /\ (reachable[v, j, g] == 1)))
        endif
      )
    )
  );

% Model with options
constraint
  forall (g in 1..nGoals) (
    forall ( i in 1..N ) (
      if i == G[g] then
        d[i, g] == 0
      elseif min(getChildDists(d, T, PO, SO, i, g)) == maxD then
        d[i, g] == maxD
      else
        d[i, g] == min(getChildDists(d, T, PO, SO, i, g)) + 1
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

constraint sum_eccentricity = sum(eccentricity);
%constraint avg_avg_distance = sum(avg_distance) / nGoals;

solve minimize sum_eccentricity;

% solve minimize sum_diameter;

output [show(PO_from[k]) ++ " " ++ show(PO_to[k]) ++ "\n"| k in po] ++
  ["d[" ++ show(n) ++ "," ++ show(g) ++ "] = " ++ show(d[n, g]) ++ "\n" | g in Goals, n in Nodes] ++
  ["eccentricity[" ++ show(G[g]) ++ "] = " ++ show(eccentricity[g]) ++ "\n" | g in Goals] ++
  ["sum(eccentricity) = " ++ show(sum_eccentricity) ++ "\n"] ++
  ["PO = " ++ show(PO) ++ "\n"] ++
  ["SO = " ++ show(SO) ++ "\n"] ++
  ["T = " ++ show(T) ++ "\n"] ++
  ["reachable" ++ show(reachable) ++ "\n"];
%  ["avg-avg distance = " ++ show(avg_avg_distance) ++ "\n"] ++
%  ["average distance[" ++ show(g) ++ "] = " ++ show(avg_distance[g]) ++ "\n" | g in Goals];
