% Version 1: we need to press ; to continue.
loop_test1 :-
  repeat,
  format('loop...~n').

% Version 2: no interruption. Use Ctrl+c to stop.
loop_test2 :-
  format('loop...~n'),
  loop_test2.

% The question is whether version 2 keeps increasing the stack?
% As version 2 is a recursive form, typical C or C++ program
% uses a lot of memory.

% Use trace to test:
% ?- trace.
% ?- loop_test2.
   %Call: (6) loop_test2 ? creep
   %Call: (7) format('loop...~n') ? creep
%loop...
   %Exit: (7) format('loop...~n') ? creep
   %Call: (7) loop_test2 ? creep
   %Call: (8) format('loop...~n') ? creep
%...
%loop...
   %Exit: (158) format('loop...~n') ? creep
   %Call: (158) loop_test2 ? creep
   %Call: (159) format('loop...~n') ? creep
% So, the stack is keep increasing.

