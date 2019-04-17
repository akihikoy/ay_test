% Integer to binary.
% cf. http://stackoverflow.com/questions/29752355/prolog-convert-integer-to-bytes

:- module(int_to_bin,
    [
      write_int32/2,
      read_int32/2
    ]).

do_write(FileName, Number) :-
    open(FileName, write, O, [all(true), encoding(octet)]),
    write_int32(O, Number),
    close(O).

do_read(FileName, Number) :-
    open(FileName, read, I, [all(true), encoding(octet)]),
    read_int32(I, Number),
    close(I).

write_int32(OS, Number) :-
  B0 is Number /\ 255,
  B1 is (Number >> 8) /\ 255,
  B2 is (Number >> 16) /\ 255,
  B3 is (Number >> 24) /\ 255,
  put_byte(OS, B0),
  put_byte(OS, B1),
  put_byte(OS, B2),
  put_byte(OS, B3).

read_int32(IS, Number) :-
  get_byte(IS, B0),
  get_byte(IS, B1),
  get_byte(IS, B2),
  get_byte(IS, B3),
  Number is B0 + B1<<8 + B2<<16 + B3<<24.

test :-
  % WARNING: user_output doesn't work with put_byte:
  % ERROR: put_byte/2: No permission to output text_stream `user_output'
  %write_int32(user_output, 3287465),
  %%read_line_to_codes(user_input, String),
  %read_int32(user_input, X),
  do_write('/tmp/byte.dat', 3287465),
  do_read('/tmp/byte.dat', X),
  write(X).

