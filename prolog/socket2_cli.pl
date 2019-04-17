% Code from http://stackoverflow.com/questions/13913182/sockets-in-swipl
% ?- create_client(localhost,20000).

% Simple protocol to communicate with Python:
% Message format = [Len|Body|Term].
%   Len: big-endian 4 byte integer that shows length of [Body|Term] (len(Body)+1).
%   Body: string that should not include Term="\n".
%   Term: terminal code "\n" (length=1).
% cf. ../../python/socket/s1_*.py

:- use_module(library(streampool)).
:- use_module('socket_util').

create_client(Host, Port) :-
  setup_call_catcher_cleanup(
      tcp_socket(Socket),
      tcp_connect(Socket, Host:Port),
      exception(_),
      tcp_close_socket(Socket)),
  setup_call_cleanup(
      tcp_open_socket(Socket, In, Out),
      chat_to_server(In, Out),
      close_connection(In, Out)).

close_connection(In, Out) :-
  format('Closing connection~n'),
  close(In, [force(true)]),
  close(Out, [force(true)]).

chat_to_server(In, Out) :-
  read_line_to_codes(user_input, Command),
  (Command=="quit"
  ->
    format('Quitting~n'),
    true
  ;
    Message= Command,
    send_msg(Out, Message),
    recv_msg(In, Reply),
    format('Reply: ~s~n', [Reply]),
    chat_to_server(In, Out)
  ).

