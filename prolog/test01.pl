% How to create a flag?

pred(A,B,C) :-
  writef('%t\n', [A]),
  writef('%t\n', [B]),
  (C=isX
    -> write('IS X\n')
    ;  write('IS NOT X\n') ).

