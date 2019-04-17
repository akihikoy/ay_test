% Figure out a way to map list elements with a predicate.
% Assume a list L=[[1,a],[2,b],[3,c]]
% and a mapping function map(In,Out) which maps a to [0,1], b to [2,3], etc.
% Goal is L to [[1,0,1],[2,2,3],...]

map(In,Out):-
  (In=a, Out=[0,1]);
  (In=b, Out=[1,2]);
  (In=c, Out=[2,3]).

apply_map(L, LM):-
  apply_map_1(L, [], Tmp),
  reverse(Tmp, LM), !.
apply_map_1([], LM, LM):- !.
apply_map_1([[Idx,Value]|Rest], Tmp, LM):-
  map(Value,Mapped),
  apply_map_1(Rest, [[Idx|Mapped]|Tmp], LM).

:- forall(apply_map([[1,a],[2,b],[3,c]], LM),     writef("%t\n",[LM])).
