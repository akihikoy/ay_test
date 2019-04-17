%http://www.learnprolognow.org/lpnpage.php?pagetype=html&pageid=lpn-htmlse26
%Exercise  6.2

palindrome([S0|Sx]):- append(Sxx,[S0],Sx), palindrome(Sxx).
palindrome([_]).
palindrome([]).

:- palindrome([r,o,t,a,t,o,r])   ,write('true\n');write('false\n').
:- palindrome([n,u,r,s,e,s,r,u,n])   ,write('true\n');write('false\n').
:- palindrome([n,o,o,n])   ,write('true\n');write('false\n').
:- palindrome([n,o,t,h,i,s])   ,write('true\n');write('false\n').
