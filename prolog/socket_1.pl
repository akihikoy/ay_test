% Code from http://prolog-adventure.blogspot.de/2012/05/prolog-and-tcp.html
%
% Use:
% $ prolog -s socket_1.pl
% ?- server(20000).
%
% Open other terminal, then:
% $ telnet localhost 20000
% Trying 127.0.0.1...
% Connected to localhost.
% Escape character is '^]'.
% hello,hello.                    %% Type something.
% You typed: hello,hello.
% Connection closed by foreign host.

:- use_module(library(streampool)).

server(Port) :-
  tcp_socket(Socket),
  tcp_bind(Socket, Port),
  tcp_listen(Socket, 5),
  tcp_open_socket(Socket, In, _Out),
  add_stream_to_pool(In, accept(Socket)),
  stream_pool_main_loop.

accept(Socket) :-
  tcp_accept(Socket, Slave, Peer),
  tcp_open_socket(Slave, In, Out),
  add_stream_to_pool(In, client(In, Out, Peer)).

client(In, Out, _Peer) :-
  read_line_to_codes(In, Command),
  close(In),
  writef('Got message: %t\n', [Command]),
  format(Out, 'You typed: ~s~n', [Command]),
  close(Out),
  delete_stream_from_pool(In).
