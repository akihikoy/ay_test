%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse26
%Exercise  6.1

accDbld([L10|L1x],A,L2):- append(A,[L10],A2), accDbld(L1x,A2,L2).
accDbld([],A,A).
doubled(L1,L2):- accDbld(L1,L1,L2).

:- doubled([a,b,c],[a,b,c,a,b,c])   ,write('true\n');write('false\n').
:- doubled([a,b,c],[a,b,c,a,b,b])   ,write('true\n');write('false\n').
:- doubled([a,b,c],T)   ,writef('%t\n',[T]).
:- doubled([aa,bb,cc],T)   ,writef('%t\n',[T]).
