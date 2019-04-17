% Example of breadth first search.
% src. https://www.cs.unm.edu/~luger/ai-final/code/PROLOG.breadth.html
% https://www.cs.unm.edu/~luger/ai-final/code/PROLOG.adts.html

empty_queue([]).
member_queue(E, S) :- member(E, S).
add_to_queue(E, [], [E]).
add_to_queue(E, [H|T], [H|Tnew]) :- add_to_queue(E, T, Tnew).
remove_from_queue(E, [E|T], T).

add_list_to_queue([], Queue, Queue).
add_list_to_queue([H|T], Queue, New_queue) :-
  add_to_queue(H, Queue, Temp_queue),
  add_list_to_queue(T, Temp_queue, New_queue).

empty_set([]).
member_set(E, S) :- member(E, S).
add_to_set(X, S, S) :- member(X, S), !.
add_to_set(X, S, [X|S]).



state_record(State, Parent, [State, Parent]).

find_path(Start, Goal) :-
  empty_queue(Empty_open),
  state_record(Start, nil, State),
  add_to_queue(State, Empty_open, Open),
  empty_set(Closed),
  path(Open, Closed, Goal).

path(Open,_,_) :- empty_queue(Open).

path(Open, Closed, Goal) :-
  remove_from_queue(Next_record, Open, _),
  state_record(State, _, Next_record),
  State = Goal,
  %writef('Next_record=%t\n',[Next_record]),
  %writef('Closed=%t\n',[Closed]),
  write('Solution path is: '), nl,
  printsolution(Next_record, Closed).

path(Open, Closed, Goal) :-
  remove_from_queue(Next_record, Open, Rest_of_open),
  (bagof(Child, moves(Next_record, Open, Closed, Child), Children);Children = []),
  add_list_to_queue(Children, Rest_of_open, New_open),
  add_to_set(Next_record, Closed, New_closed),
  path(New_open, New_closed, Goal),!.

moves(State_record, Open, Closed, Child_record) :-
  state_record(State, _, State_record),
  %mov(State, Next),
  possible_actions(State, Action),
  next_state(State, Action, Next),
  % not (unsafe(Next)),
  state_record(Next, _, Test),
  not(member_queue(Test, Open)),
  not(member_set(Test, Closed)),
  state_record(Next, State, Child_record).

printsolution(State_record, _):-
  state_record(State,nil, State_record),
  writef(' %t', [State]).
printsolution(State_record, Closed) :-
  state_record(State, Parent, State_record),
  state_record(Parent, _, Parent_record),
  member(Parent_record, Closed),
  printsolution(Parent_record, Closed),
  writef(' %t', [State]).










% Find a path from Start to Goal.
%find_path(Start, Goal, Result) :-
  %%search(Start, Goal, [Start], Q),
  %%reverse(Q, Result).                   %Result is left-to-right
  %search(Start, Goal, [Start], Result).  %Result is right-to-left

%search(State, Goal, PastSeq, Result) :-
  %State=Goal,
  %Result=[goal|PastSeq],
  %!.
%search(State, Goal, PastSeq, Result) :-
  %possible_actions(State, Action),
  %next_state(State, Action, Next),
  %%\+member(Next, PastSeq),
  %%search(Next, Goal, [Next|PastSeq], Result).
  %(\+member(Next, PastSeq)
    %->
      %search(Next, Goal, [Next,Action|PastSeq], Result)
    %;
      %Result= [loop,Next,Action|PastSeq]
    %).


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
:- find_path(a, e).

