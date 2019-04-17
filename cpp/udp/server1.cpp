
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cstring>
#include <sstream>

namespace server
{
  int                 SockFD (-1);
  struct sockaddr_in  ClientAddress;
  socklen_t           CALength;

  void Setup (int port)
  {
    struct sockaddr_in addr;

    SockFD = socket(AF_INET, SOCK_DGRAM, 0);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(SockFD, (struct sockaddr *)&addr, sizeof(addr));

    // ClientAddress.sin_family = AF_INET;
    // ClientAddress.sin_port = htons(port);
    // inet_aton("127.0.0.1", &ClientAddress.sin_addr);
  }

  inline ssize_t ReceiveFromClient (void *buffer, int buffer_length)
  {
    CALength= sizeof(sockaddr_in);
    ssize_t n= recvfrom(SockFD, buffer, buffer_length, 0, (struct sockaddr *)&ClientAddress, &CALength);
    // LMESSAGE("reveiced "<<n<<"[byte] data from "<<inet_ntoa(ClientAddress.sin_addr));
      //!\todo <b>FIXME: error check for ClientAddress, CALength </b>
    return n;
  }

  template <typename t_data>
  inline ssize_t SendToClient (const t_data &v_data, bool broad=false)
  {
    if(!broad)
      return sendto (SockFD, &v_data, sizeof(t_data), 0, (struct sockaddr *)&ClientAddress, sizeof(ClientAddress));
    else
      return sendto (SockFD, &v_data, sizeof(t_data), 0, NULL, 0);
  }

  inline ssize_t SendToClient (const char *v_data, int v_length, bool broad=false)
  {
    if(!broad)
      return sendto (SockFD, v_data, v_length, 0, (struct sockaddr *)&ClientAddress, sizeof(ClientAddress));
    else
      return sendto (SockFD, v_data, v_length, 0, NULL, 0);
  }

}

#include <iostream>

using namespace server;
using namespace std;

int main(int argc,char **argv)
{
  int port(9877);
  if(argc>=2)
  {
    stringstream ss(argv[1]);
    ss>>port;
  }
  Setup (port);

  char data[256];
  int len=ReceiveFromClient(data,sizeof(data)-1);
  if(len>0)
  {
    data[len]='\0';
    cout<<data<<endl;
  }

  string command;
  while(true)
  {
    cout<<" > "<<flush;
    cin>>command;
    SendToClient(command.c_str(),command.length());
    if(command=="quit")
    {
      break;
    }
  }
  return 0;
}
