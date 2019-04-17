%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse11
%Exercise  3.2

directlyIn(olga,katarina).
directlyIn(natasha,olga).
directlyIn(irina,natasha).
in(X,Y):- directlyIn(X,Y).
in(X,Y):- directlyIn(X,Z), in(Z,Y).
