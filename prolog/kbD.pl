%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse11
%Exercise  3.1

child(jiro, taro).
child(sabu, jiro).

descend(X,Y):-child(X,Y).
descend(X,Y):-child(X,Z),descend(Z,Y).

%descend(X,Y):-child(X,Y).
%descend(X,Y):-descend(X,Z),descend(Z,Y).
