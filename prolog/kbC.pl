% Find all:
mem(X, [X|_]). % x corresponds with a head of a list.
mem(X, [_|Y]) :- mem(X, Y). % otherwise.

% Find first:
mem1(X, [X|_]):-!.
mem1(X, [_|Y]) :- mem1(X, Y). % otherwise.
