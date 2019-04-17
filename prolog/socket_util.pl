:- module(socket_util,
    [
      write_int32/2,
      read_int32/2,
      send_msg/2,
      recv_msg/2
    ]).

:- use_module(library(readutil)).

write_int32(OS, Number) :-
  B0 is Number /\ 255,
  B1 is (Number >> 8) /\ 255,
  B2 is (Number >> 16) /\ 255,
  B3 is (Number >> 24) /\ 255,
  put_byte(OS, B3),
  put_byte(OS, B2),
  put_byte(OS, B1),
  put_byte(OS, B0).

read_int32(IS, Number) :-
  get_byte(IS, B3),
  get_byte(IS, B2),
  get_byte(IS, B1),
  get_byte(IS, B0),
  Number is B0 + B1<<8 + B2<<16 + B3<<24.

% Msg is a string (list of integers).
send_msg(Out, Msg) :-
  length(Msg, Len),
  Len2 is Len+1,  % +1 is for the terminal code.
%format('~q ~s ~q ~q ~n',[Msg,Msg,Len,Len2]),
  write_int32(Out, Len2),
  format(Out, '~s~n', [Msg]),
  flush_output(Out).

% Receive Msg.
recv_msg(In, Msg) :-
  %read_line_to_codes(In, HdMsg),
%((length(HdMsg,Len),Len>4)
%->
  %[_,_,_,_|Msg]=HdMsg  % Remove header
  %%[H1,H2,H3,H4|Msg]=HdMsg,  % Remove header
  %%format('~q~n',[[H1,H2,H3,H4]]).
%;
%Msg=HdMsg).
  %recv_msg_1(In, [], Header, 4),
  read_int32(In, Len),
  %format('receiving ~q letters~n',[Len]),
  read_line_to_codes(In, Msg).

%recv_msg_1(In, Msg, Msg, N) :- length(Msg,Len), Len>=N, !.
%recv_msg_1(In, PrevMsg, Msg, N) :-
  %read_line_to_codes(In, TmpMsg),
  %append(PrevMsg, TmpMsg, NewMsg),
  %recv_msg(In, NewMsg, Msg, N).

