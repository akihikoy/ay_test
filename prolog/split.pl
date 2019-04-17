
split([],[],[]).
split([X|L],[X|L1],L2) :-
  X >= 0,
  !,
  split(L,L1,L2).
split([X|L],L1,[X|L2]) :-
  split(L,L1,L2).

:- L=[1,2,-3,0,2,-3,-9], split(L,L1,L2), format('~q --> ~q, ~q', [L,L1,L2]).
