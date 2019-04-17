%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse11
%Exercise  3.3

directTrain(saarbruecken,dudweiler).
directTrain(forbach,saarbruecken).
directTrain(freyming,forbach).
directTrain(stAvold,freyming).
directTrain(fahlquemont,stAvold).
directTrain(metz,fahlquemont).
directTrain(nancy,metz).
directTrain(tokyo,kyoto).

%%Doesn't consider the opposite direction:
%travelFromTo(X,Y):- directTrain(X,Y).
%travelFromTo(X,Y):- directTrain(X,Z), travelFromTo(Z,Y).

travelFromTo(X,Y):- X=Y.
travelFromTo(X,Y):- directTrain(X,Y); directTrain(Y,X).
travelFromTo(X,Y):- (directTrain(X,Z), travelFromTo(Z,Y)); (directTrain(Y,Z), travelFromTo(Z,X)).

%%This gives only a single answer.
%travelFromTo(X,Y):- directTrain(X,Y), !;
                    %directTrain(Y,X), !.
%travelFromTo(X,Y):- (directTrain(X,Z), travelFromTo(Z,Y)), !;
                    %(directTrain(Y,Z), travelFromTo(Z,X)), !.

%This gives only a single answer.
%This doesn't stop for travelFromTo(tokyo,metz).
%travelFromTo(X,Y):- X=Y, !.
%travelFromTo(X,Y):- directTrain(X,Y), !.
%travelFromTo(X,Y):- directTrain(X,Z), travelFromTo(Z,Y), !.
%travelFromTo(X,Y):- travelFromTo(Y,X).

%?- travelFromTo(X,Y).
%?- travelFromTo(metz,nancy).
%?- travelFromTo(kyoto,kyoto).
%?- travelFromTo(tokyo,metz).
