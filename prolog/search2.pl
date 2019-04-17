%https://www.cpp.edu/~jrfisher/www/prolog_tutorial/2_16.html
%2.16 Search

%A general search method:

solve(P) :-
  start(Start),
  search(Start,[Start],Q),
  reverse(Q,P).

search(S,P,P) :- goal(S), !.         /* done                  */
search(S,Visited,P) :-
  next_state(S,Nxt),              /* generate next state   */
  safe_state(Nxt),                /* check safety          */
  no_loop(Nxt,Visited),           /* check for loop        */
  search(Nxt,[Nxt|Visited],P).    /* continue searching... */

no_loop(Nxt,Visited) :-
  \+member(Nxt,Visited).


%Define a domain:

start([]).
goal(S) :- length(S,4).
%goal(S) :- length(S,8).

next_state(S,[C|S]) :-
  member(C,[1,2,3,4]),
  %member(C,[1,2,3,4,5,6,7,8]),
  \+member(C,S).

safe_state([C|S]) :-
  length(S,L),
  Sum is C+L+1, Diff is C-L-1,
  safe_state(S,Sum,Diff).


safe_state([],_,_) :- !.
safe_state([F|R],Sm,Df) :-
  length(R,L),
  X is F+L+1,
  X \= Sm,
  Y is F-L-1,
  Y \= Df,
  safe_state(R,Sm,Df).

:- findall(P,(solve(P)          ,writef('%t\n',[P])),_).
