%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse25

naiverev([],[]).
naiverev([H|T],R):-  naiverev(T,RevT),  append(RevT,[H],R).

accRev([H|T],A,R):-  accRev(T,[H|A],R).
accRev([],A,A).
rev(L,R):-  accRev(L,[],R).

:- naiverev([a,b,c,[d,e],f],T)   ,writef('%t\n',[T]).
:- rev([a,b,c,[d,e],f],T)   ,writef('%t\n',[T]).
