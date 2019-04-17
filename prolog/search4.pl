% A general search method.
% NOTE: This is a depth-first search.
% Maybe breadth-first search is preferable to me?
% Or implement A* search.

% Find a path from Start to Goal.
find_path(Start, Goal, Result) :-
  %search(Start, Goal, [Start], Q),
  %reverse(Q, Result).                   %Result is left-to-right
  search(Start, Goal, [Start], Result).  %Result is right-to-left

search(State, Goal, PastSeq, Result) :-
  State=Goal,
  Result=[goal|PastSeq],
  !.
search(State, Goal, PastSeq, Result) :-
  possible_actions(State, Action),
  next_state(State, Action, Next),
  %\+member(Next, PastSeq),
  %search(Next, Goal, [Next|PastSeq], Result).
  (\+member(Next, PastSeq)
    ->
      search(Next, Goal, [Next,Action|PastSeq], Result)
    ;
      Result= [loop,Next,Action|PastSeq]
    ).

% Generate a graph from Start to Goal.
% The mission is remove "dead-end loops", and reshape the graph.
make_graph(Start, Goal, Graph) :-
  find_path(Start, Goal, Result), Graph=Result.  %Of course this is not a solution....FIXME!
  %TODO: This is incomplete!!


%Define a domain:

% states: {a,b,c,d,e,f,g}
% possible action: {a:{jb,jc}, b:{jc,jd}, c:{je,jf,ja}, d:{je}, e:{}, f:{jg}, g:{jf}}
% action effect: jX: jump to state X
% start: a
% goal: e

possible_actions(State, Action):-
  (State=a,member(Action,[jb,jc]));
  (State=b,member(Action,[jc,jd]));
  (State=c,member(Action,[je,jf,ja]));
  (State=d,member(Action,[je]));
  (State=e,member(Action,[]));
  (State=f,member(Action,[jg]));
  (State=g,member(Action,[jf])).

next_state(State, Action, Next):-
  member(State,[a,b,c,d,e,f,g]),
  atom_concat(j,Next,Action).


% Test:
:- findall(Result,(find_path(a, e, Result)          ,writef('%t\n',[Result])),_).

