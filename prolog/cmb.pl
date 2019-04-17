% Generate all combinations.
% Assume we have a knowledge like:
%   L=[[a,1,2,3,4], [b,5,6,7], [c,8,9]]
% means "a" takes one of [1,2,3,4], "b" takes...
% Goal is generate all combinations like
%   [[a,1],[b,5],[c,8]], [[a,1],[b,5],[c,9]], ...

cmb(C):-
  KB=[[a,1,2,3,4], [b,5,6,7], [c,8,9]],
  cmb(KB, [], C).

cmb([], C, C):- !.
cmb(KB, C, CC):-
  [First|KBRest]=KB,
  [A|L]=First,
  member(M,L),
  cmb(KBRest, [[A,M]|C], CC).


:- forall(cmb(C),     writef("%t\n",[C])).
