%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse16
%Exercise  4.4

%swap12(L1,L2):- [X1,X2|Lo]=L1, [X2,X1|Lo]=L2.

%Equivalent:
swap12([X1,X2|Lo],[X2,X1|Lo]).

:- swap12([a,b,c,d],[a,b,c,d])  ,write('true\n');write('false\n').
:- swap12([a,b,c,d],[b,a,c,d])  ,write('true\n');write('false\n').
:- swap12([a,b,c,d],T)  ,writef('%t\n',[T]).
:- swap12([[a,b],c,d],T)  ,writef('%t\n',[T]).
