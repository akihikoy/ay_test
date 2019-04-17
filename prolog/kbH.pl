%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse16
%Exercise  4.3

%second(X,List):- [_,X|_]=List.

%Equivalent:
second(X,[_,X|_]).

:- second(a,[a,b,c,d])  ,write('true\n');write('false\n').
:- second(b,[a,b,c,d])  ,write('true\n');write('false\n').
:- second(c,[a,b,c,d])  ,write('true\n');write('false\n').
:- second(b,[a,[b,c],d])  ,write('true\n');write('false\n').
:- second([b,c],[a,[b,c],d])  ,write('true\n');write('false\n').
