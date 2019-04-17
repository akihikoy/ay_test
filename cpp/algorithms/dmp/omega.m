function y=f(omega)
  y= omega*cos(omega)+2;
endfunction

fsolve("f",12.0)
