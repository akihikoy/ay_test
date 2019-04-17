%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse16
%Exercise  4.5

tran(eins,one).
tran(zwei,two).
tran(drei,three).
tran(vier,four).
tran(fuenf,five).
tran(sechs,six).
tran(sieben,seven).
tran(acht,eight).
tran(neun,nine).

%listtran(G,E):- []=G, []=E.
%listtran(G,E):- [G1|Gx]=G, [E1|Ex]=E, tran(G1,E1), listtran(Gx,Ex).

%Equivalent:
listtran([],[]).
listtran([G1|Gx],[E1|Ex]):- tran(G1,E1), listtran(Gx,Ex).

:- listtran([eins,zwei,drei],[one,two,three])   ,write('true\n');write('false\n').
:- listtran([eins,zwei,drei], X)    ,writef('%t\n',[X]).
:- listtran(X, [one,two,three])    ,writef('%t\n',[X]).
:- listtran([neun,fuenf,sechs,acht,vier], X)    ,writef('%t\n',[X]).
