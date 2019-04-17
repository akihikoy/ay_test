%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse22
%Exercise  5.3

%addone([],X):- X=[].
%addone(List,X):- [L1|Lx]=List, addone(Lx,Y), L2 is L1+1, X=[L2|Y].

%Equivalent:
addone([],[]).
addone([L1|Lx],X):- addone(Lx,Y), L2 is L1+1, X=[L2|Y].

:- addone([1,2,3], X)    ,writef('%t\n',[X]).
:- addone([-8,2,-2,0], X)    ,writef('%t\n',[X]).
