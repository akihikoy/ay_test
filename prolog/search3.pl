%https://www.cpp.edu/~jrfisher/www/prolog_tutorial/5_1.html
%5.1 The A* algorithm in Prolog

%A general search method:

%%%     A* Algorithm
%%%
%%%
%%%     Nodes have form    S=>D=>F=>A
%%%            where S describes the state or configuration
%%%                  D is the depth of the node
%%%                  F is the evaluation function value
%%%                  A is the ancestor list for the node

:- op(400,yfx,'=>').    /* Node builder notation */

solve(Start,Soln) :- f_function(Start,0,F),
                     search([Start=>0=>F=>[]],S),
                     reverse(S,Soln).

f_function(State,D,F) :- h_function(State,H),
                         F is D + H.

search([State=>_=>_=>Soln | _], Soln) :- goal(State).
search([B|R],S) :- expand(B, Children),
                   insert_all(Children, R, NewOpen),
                   search(NewOpen,S).

expand(State=>D=>_=>A, All_My_Children) :-
         bagof(Child=>D1=>F=>[Move|A],
               ( D1 is D + 1,
                 move(State,Child,Move),
                 f_function(Child,D1,F) ) ,
               All_My_Children).

insert_all([F|R],Open1,Open3) :- insert(F,Open1,Open2),
                                 insert_all(R,Open2,Open3).
insert_all([],Open,Open).

insert(B,Open,Open) :- repeat_node(B,Open), ! .
insert(B,[C|R],[B,C|R]) :- cheaper(B,C), ! .
insert(B,[B1|R],[B1|S]) :- insert(B,R,S), !.
insert(B,[],[B]).

repeat_node(P=>_=>_=>_, [P=>_=>_=>_|_]).

cheaper( _=>_=>H1=>_ , _=>_=>H2=>_ ) :- H1 <  H2.


%Define a domain:
%We need to define goal, move, and h_function for each domain
%NOTE: under construction...

%start([]).
goal(S) :- length(S,4).
%goal(S) :- length(S,8).

move(State,Child,Move) :-
  member(Child,[1,2,3,4]),
  \+member(Child,State),
  safe_state()

%next_state(S,[C|S]) :-
  %member(C,[1,2,3,4]),
  %%member(C,[1,2,3,4,5,6,7,8]),
  %\+member(C,S).

safe_state([C|S]) :-
  length(S,L),
  Sum is C+L+1, Diff is C-L-1,
  safe_state(S,Sum,Diff).


%safe_state([],_,_) :- !.
%safe_state([F|R],Sm,Df) :-
  %length(R,L),
  %X is F+L+1,
  %X \= Sm,
  %Y is F-L-1,
  %Y \= Df,
  %safe_state(R,Sm,Df).

%:- findall(P,(solve(P)          ,writef('%t\n',[P])),_).

