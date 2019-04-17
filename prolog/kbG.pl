%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse11
%Exercise  3.5

%leaf(L).
%tree(B1,B2).

%TEST:
%swap(T1,T2):- B1=leaf(L1), B2=leaf(L2), T1=tree(B1,B2), T2=tree(leaf(L2),leaf(L1)).

%TEST:
%swap(T1,T2):- T1=tree(leaf(L1),leaf(L2)), T2=tree(leaf(L2),leaf(L1)).
%swap(T1,T2):- T1=tree(tree(leaf(L1),leaf(L2)),leaf(L3)), T2=tree(leaf(L3),tree(leaf(L2),leaf(L1))).
%swap(T1,T2):- T1=tree(leaf(L1),tree(leaf(L2),leaf(L3))), T2=tree(tree(leaf(L3),leaf(L2)),leaf(L1)).

%WORKS:
%swap(T1,T2):- T1=tree(leaf(L1),leaf(L2)), T2=tree(leaf(L2),leaf(L1)).
%swap(T1,T2):- T1=tree(tree(B1,B2),leaf(L3)), T2=tree(leaf(L3),T3), swap(tree(B1,B2),T3).
%swap(T1,T2):- T1=tree(leaf(L1),tree(B2,B3)), T2=tree(T3,leaf(L1)), swap(tree(B2,B3),T3).
%swap(T1,T2):- T1=tree(tree(B1,B2),tree(B3,B4)), T2=tree(T4,T3), swap(tree(B1,B2),T3),swap(tree(B3,B4),T4).

%Equivalent:
%swap(tree(leaf(L1),leaf(L2)), tree(leaf(L2),leaf(L1))).
%swap(tree(tree(B1,B2),leaf(L3)), tree(leaf(L3),T3)):- swap(tree(B1,B2),T3).
%swap(tree(leaf(L1),tree(B2,B3)), tree(T3,leaf(L1))):- swap(tree(B2,B3),T3).
%swap(tree(tree(B1,B2),tree(B3,B4)), tree(T4,T3)):- swap(tree(B1,B2),T3),swap(tree(B3,B4),T4).

%Equivalent:
swap(T1,T2):-
    ( T1=tree(leaf(L1),leaf(L2)), T2=tree(leaf(L2),leaf(L1)) );
    ( T1=tree(tree(B1,B2),leaf(L3)), T2=tree(leaf(L3),T3), swap(tree(B1,B2),T3) );
    ( T1=tree(leaf(L1),tree(B2,B3)), T2=tree(T3,leaf(L1)), swap(tree(B2,B3),T3) );
    ( T1=tree(tree(B1,B2),tree(B3,B4)), T2=tree(T4,T3), swap(tree(B1,B2),T3),swap(tree(B3,B4),T4) ).

%[kbG].
:- swap(tree(leaf(1),leaf(2)), T), writef('%t\n',[T]).
:- swap(tree(leaf(1),tree(leaf(2),leaf(3))), T), writef('%t\n',[T]).
:- swap(tree(tree(leaf(1),leaf(2)),leaf(3)), T), writef('%t\n',[T]).
:- swap(tree(tree(leaf(1),leaf(2)),tree(leaf(3),leaf(4))), T), writef('%t\n',[T]).
:- swap(tree(tree(leaf(1),leaf(2)),tree(tree(leaf(3),leaf(4)),leaf(5))), T), writef('%t\n',[T]).
:- swap(tree(tree(tree(tree(leaf(3),leaf(4)),leaf(5)),tree(tree(tree(tree(leaf(3),leaf(4)),leaf(5)),leaf(4)),leaf(5))),tree(tree(leaf(3),tree(tree(leaf(3),leaf(4)),leaf(5))),tree(tree(tree(tree(leaf(3),leaf(4)),leaf(5)),leaf(4)),leaf(5)))), T), writef('%t\n',[T]).

:- swap(tree(tree(tree(tree(leaf(3),leaf(4)),leaf(5)),tree(tree(tree(tree(leaf(3),leaf(4)),leaf(5)),leaf(4)),leaf(5))),tree(tree(leaf(3),tree(tree(leaf(3),leaf(4)),leaf(5))),tree(tree(tree(tree(leaf(3),leaf(4)),leaf(5)),leaf(4)),leaf(5)))), _).

:- writef('END\n').
%:- halt.
