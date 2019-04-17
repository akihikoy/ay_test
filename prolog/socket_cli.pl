% Code from http://stackoverflow.com/questions/13913182/sockets-in-swipl
% ?- create_client(localhost,20000).

:- use_module(library(streampool)).

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
  %read(Command),
  read_line_to_codes(user_input, Command),
  (atom_codes('quit',Command)
    %Command == quit
  ->
    format('Quitting~n'),
    true
  ;
    %format(Out, '~q .~n', [Command]),
    format(Out, '~s~n', [Command]),
    flush_output(Out),
    %read(In, Reply),
    %format('Reply: ~q.~n', [Reply]),
    read_line_to_codes(In, Reply),
    format('Reply: ~s~n', [Reply]),
    chat_to_server(In, Out)
  ).

