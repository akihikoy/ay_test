%https://www.cpp.edu/~jrfisher/www/prolog_tutorial/2_11.html
%2.11 Chess queens challenge puzzle
:- use_module(library(lists)).

solve(Domain,P):-
  lists:perm(Domain,P),
  combine(Domain,P,S,D),
  all_diff(S),
  all_diff(D).

combine([X1|X],[Y1|Y],[S1|S],[D1|D]):-
  S1 is X1 + Y1,
  D1 is X1 - Y1,
  combine(X,Y,S,D).
combine([],[],[],[]).

all_diff([X|Y]):- \+member(X,Y), all_diff(Y).
all_diff([_]).

:- findall(P,(solve([1,2,3,4],P)          ,writef('%t\n',[P])),_).
:- findall(P,(solve([1,2,3,4,5,6,7,8],P)  ,writef('%t\n',[P])),_).
