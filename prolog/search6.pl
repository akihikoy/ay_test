% Example of breadth first search.
% src. http://www.cs.nott.ac.uk/~pszbsl/G52APT/slides/10-Breadth-first-search.pdf
% p.23 Breadth first search with closed list 3

% BFS with no loop detection:
bfs(Start,Goal,Result) :-
  bfs_a(Goal,[n(Start,[])],R),
  reverse(R,Result).
bfs_a(Goal, [n(State,Result)|_], Result) :-
  State=Goal.
bfs_a(Goal, [n(State,ActSeq)|RestOPaths], Result) :-
  findall(n(Next,[Action|ActSeq]),
      ( possible_actions(State, Action),
        next_state(State, Action, Next) ),
      NextOPaths),
  append(RestOPaths, NextOPaths, OPaths),
  bfs_a(Goal, OPaths, Result).


% bfs(?initial_state, ?goal_state, ?solution)
% OPaths: Paths to be opened, e.g. [n(d,[jd,jb]),n(e,[je,jc]),n(f,[jf,jc])]
%       The top of OPaths is opened.
find_path(Start, Goal, Result) :-
  %bfs_d(Goal, [n(Start,[])], [], R),
  %bfs_d2(Goal, [n(Start,[])], [], R),
  bfs_d3(Goal, [[Start]], [], R),
  %reverse(R, Result).
  Result=R.

% Similar to member/2, but stops when an element is detected.
member1(Elem,List) :- member(Elem,List), !.

bfs_d(Goal, [n(State,Result)|RestOPaths], _, Result) :-
writef('*-DEBUG:OPaths: %t\n',[[n(State,Result)|RestOPaths]]),
  State=Goal.
bfs_d(Goal, [n(State,ActSeq)|RestOPaths], Closed, Result) :-
  length(ActSeq, L),
writef('1-DEBUG:Closed: %t\n',[Closed]),
writef('2-DEBUG:OPaths: %t\n',[[n(State,ActSeq)|RestOPaths]]),
  findall(n(Next,[Action|ActSeq]),
      ( possible_actions(State, Action),
        next_state(State, Action, Next),
        \+ (member1(n(Next,ActSeq2),RestOPaths), length(ActSeq2,L)),  % Condition for shortest?
        \+ member1(Next, Closed) ),  % Condition to remove loop
      NextOPaths),
  append(RestOPaths, NextOPaths, OPaths),
writef('3-DEBUG:OPaths: %t\n',[OPaths]),
  bfs_d(Goal, OPaths, [State|Closed], Result).

% Modified so that we list-up all possible solutions (not only shortest) including loops.
bfs_d2(Goal, [n(State,ActSeq)|_], _, Result) :-
%bfs_d2(Goal, [n(State,ActSeq)|RestOPaths], _, Result) :-
%writef('*-DEBUG:OPaths: %t\n',[[n(State,Result)|RestOPaths]]),
  State=Goal,
  Result=[goal|ActSeq].
bfs_d2(_, [n(loop,ActSeq)|_], _, Result) :-
%bfs_d2(Goal, [n(loop,ActSeq)|RestOPaths], _, Result) :-
%writef('*-DEBUG:OPaths: %t\n',[[n(loop,Result)|RestOPaths]]),
  Result=[loop|ActSeq].
bfs_d2(Goal, [n(State,ActSeq)|RestOPaths], Closed, Result) :-
  %length(ActSeq, L),
%writef('1-DEBUG:Closed: %t\n',[Closed]),
%writef('2-DEBUG:OPaths: %t\n',[[n(State,ActSeq)|RestOPaths]]),
  findall(n(Next,[Action|ActSeq]),
      ( possible_actions(State, Action),
        next_state(State, Action, Next) ),
      PossibleOPaths),
  findall(n(Next,[Action|ActSeq]),
      ( member(n(Next,[Action|ActSeq]), PossibleOPaths),
        %\+ (member1(n(Next,ActSeq2),RestOPaths), length(ActSeq2,L)),  % Condition for shortest?
        \+ member1(Next, Closed) ),  % Condition to remove loop <-- TODO: KEEP THIS.
      NextOPaths),
  findall(n(loop,[Action|ActSeq]),
      ( member(n(Next,[Action|ActSeq]), PossibleOPaths),
        member1(Next, Closed) ),
      LoopPaths),
  append(RestOPaths, LoopPaths, Tmp),
  append(Tmp, NextOPaths, OPaths),
%writef('3-DEBUG:OPaths: %t\n',[OPaths]),
  bfs_d2(Goal, OPaths, [State|Closed], Result).

% Modified so that we list-up all possible solutions (not only shortest) including loops.
% We return state-action sequence.
bfs_d3(Goal, [StActSeq|_], _, Result) :-
%bfs_d3(Goal, [StActSeq|RestOPaths], _, Result) :-
%writef('*-DEBUG:OPaths: %t\n',[[StActSeq|RestOPaths]]),
  [State|_]= StActSeq,
  State=Goal,
  Result=[goal|StActSeq].
bfs_d3(_, [[loop|StActSeq]|_], _, Result) :-
%bfs_d3(Goal, [[loop|StActSeq]|RestOPaths], _, Result) :-
%writef('*-DEBUG:OPaths: %t\n',[[[loop|StActSeq]|RestOPaths]]),
  Result=[loop|StActSeq].
bfs_d3(Goal, [StActSeq|RestOPaths], Closed, Result) :-
  [State|_]= StActSeq,
  %length(StActSeq, L),
%writef('1-DEBUG:Closed: %t\n',[Closed]),
%writef('2-DEBUG:OPaths: %t\n',[[StActSeq|RestOPaths]]),
  findall([Next,Action|StActSeq],
      ( possible_actions(State, Action),
        next_state(State, Action, Next) ),
      PossibleOPaths),
  findall([Next,Action|StActSeq],
      ( member([Next,Action|StActSeq], PossibleOPaths),
        %\+ (member1(Next,ActSeq2,RestOPaths), length(ActSeq2,L)),  % Condition for shortest?
        \+ member1(Next, Closed) ),  % Condition to remove loop <-- TODO: KEEP THIS.
      NextOPaths),
  findall([loop,Next,Action|StActSeq],
      ( member([Next,Action|StActSeq], PossibleOPaths),
        member1(Next, Closed) ),
      LoopPaths),
%writef('2.2-DEBUG: %t\n',[PossibleOPaths]),
%writef('2.5-DEBUG: %t :: %t\n',[NextOPaths,LoopPaths]),
  append(RestOPaths, LoopPaths, Tmp),
  append(Tmp, NextOPaths, OPaths),
%writef('3-DEBUG:OPaths: %t\n',[OPaths]),
  bfs_d3(Goal, OPaths, [State|Closed], Result).


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

