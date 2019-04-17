next1(X,Y):-
  (X=a,Y=b);
  (X=b,Y=c);
  (X=c,Y=a).

next2(X,Y):-
  (X=a,Y=b);
  (X=b,Y=c);
  (X=c,Y=d);
  (X=d,Y=a).

%propagate(_,Result):- length(Result,5),!.
%propagate(Start,Result):-
  %next(Start,Next),
  %propagate(Next,[Next|Result]).

propagate(Start,Result):-
  propagate(Start,[Start],R),
  reverse(R,Result).
propagate(_,Tmp,Tmp):- length(Tmp,5),!.
propagate(Start,Tmp,Result):-
  next1(Start,Next),
  propagate(Next,[Next|Tmp],Result).

% Usage:
%?- propagate(a,Result).


% Generalized version: We want to switch next1 to next2 by an argument.
propagateG(Start,PNext,Result):-
  propagateG(Start,PNext,[Start],R),
  reverse(R,Result).
propagateG(_,_,Tmp,Tmp):- length(Tmp,5),!.
propagateG(Start,PNext,Tmp,Result):-
  next(PNext,Start,Next),
  propagateG(Next,PNext,[Next|Tmp],Result).

% Now users can define any next functions.
next(next1,Start,Next):- next1(Start,Next).
next(next2,Start,Next):- next2(Start,Next).

% Usage:
%?- propagateG(a,next1,Result).

