% Code from http://stackoverflow.com/questions/13913182/sockets-in-swipl
% ?- create_server(20000).

:- use_module(library(socket)).

create_server(Port) :-
  tcp_socket(Socket),
  tcp_bind(Socket, Port),
  tcp_listen(Socket, 5),
  %tcp_open_socket(Socket, AcceptFd, _),
  tcp_open_socket(Socket, AcceptFd),
  %dispatch(AcceptFd).
  format('Waiting for a connection~n'),
  tcp_accept(AcceptFd, Socket2, _),
  setup_call_cleanup(
      tcp_open_socket(Socket2, In, Out),
      handle_service(In, Out),
      close_connection(AcceptFd, In, Out)).

%dispatch(AcceptFd) :-
  %tcp_accept(AcceptFd, Socket, Peer),
  %thread_create(process_client(Socket, Peer), _,
                %[ detached(true)
                %]),
  %dispatch(AcceptFd).

%process_client(Socket, _Peer) :-
  %setup_call_cleanup(tcp_open_socket(Socket, In, Out),
                      %handle_service(In, Out),
                      %close_connection(In, Out)).

%close_connection(In, Out) :-
  %close(In, [force(true)]),
  %close(Out, [force(true)]).

%dispatch(AcceptFd) :-
  %tcp_accept(AcceptFd, Socket, Peer),
  %setup_call_cleanup(tcp_open_socket(Socket, In, Out),
                      %handle_service(In, Out),
                      %close_connection(AcceptFd, In, Out)).

close_connection(Socket, In, Out) :-
  format('Closing connection~n'),
  close(In, [force(true)]),
  close(Out, [force(true)]),
  tcp_close_socket(Socket).

handle_service(In, Out) :-
  %read(In, Command),
  read_line_to_codes(In, Command),
  %writeln(Command),
  (Command==end_of_file
  ->
    format('Client has disconnected~n'),
    true
    %!,fail
  ;
    format('Got message:: ~s~n', [Command]),
    %format(Out, 'seen(~q).~n', [Command]),
    format(Out, 'You said: ~s~n', [Command]),
    flush_output(Out),
    handle_service(In, Out)
  ).

%call_test(test,Term):-Term = 'really test'.
%call_test(validate(teste),Term):-
    %String = "validate(test)",
    %string_to_list(String,List),
    %read_from_chars(List,Stringf),
    %writeln(Stringf),
    %Term = 'foo'.
