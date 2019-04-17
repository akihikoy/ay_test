% Code from http://stackoverflow.com/questions/13913182/sockets-in-swipl
% ?- create_server(20000).

% Simple protocol to communicate with Python:
% Message format = [Len|Body|Term].
%   Len: big-endian 4 byte integer that shows length of [Body|Term] (len(Body)+1).
%   Body: string that should not include Term="\n".
%   Term: terminal code "\n" (length=1).
% cf. ../../python/socket/s1_*.py

:- use_module(library(socket)).
:- use_module('socket_util').

create_server(Port) :-
  tcp_socket(Socket),
  tcp_bind(Socket, Port),
  tcp_listen(Socket, 5),
  tcp_open_socket(Socket, AcceptFd),
  format('Waiting for a connection~n'),
  tcp_accept(AcceptFd, Socket2, _),
  format('Connected~n'),
  setup_call_cleanup(
      tcp_open_socket(Socket2, In, Out),
      handle_service(In, Out),
      close_connection(AcceptFd, In, Out)).

close_connection(Socket, In, Out) :-
  format('Closing connection~n'),
  close(In, [force(true)]),
  close(Out, [force(true)]),
  tcp_close_socket(Socket).

handle_service(In, Out) :-
  recv_msg(In, Command),
  (Command==end_of_file
  ->
    format('Client has disconnected~n'),
    true
  ;
    format('Got message:: ~s~n', [Command]),
    append("You said: ", Command, Message),
    send_msg(Out, Message),
    handle_service(In, Out)
  ).

