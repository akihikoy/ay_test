
namespace client
{
  void Setup (const char *address, int port);
  void SetConnectionErrorHandler (void (*f)(void));
  void ClearServerQueue (void);
}

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <lora/common.h>

namespace client
{
  int                 SockFD (-1);
  struct sockaddr_in  ServerAddress;
  void (*ConnectionErrorHandler)(void)= NULL;

  void SetTimeout (int sec, int usec)
  {
    struct timeval tv;
    tv.tv_sec= sec;
    tv.tv_usec= usec;
    setsockopt(SockFD, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }

  void SetTimeout (void)
  {
    /*TEST*/SetTimeout(10,0);
  }

  void Setup (const char *address, int port)
  {
    SockFD = socket(AF_INET, SOCK_DGRAM, 0);

    ServerAddress.sin_family = AF_INET;
    ServerAddress.sin_port = htons(port);
    inet_aton(address, &ServerAddress.sin_addr);

    SetTimeout();
  }

  void SetConnectionErrorHandler (void (*f)(void))
  {
    ConnectionErrorHandler= f;
  }

  void ClearServerQueue (void)
  {
    char buf[2048];
    struct sockaddr_in  from_addr;
    socklen_t  fa_len= sizeof(sockaddr_in);
    SetTimeout(0,1000);
    while(1)
    {
      ssize_t n= recvfrom(SockFD, buf, sizeof(buf), 0, (struct sockaddr *)&from_addr, &fa_len);
      LDEBUG("skip "<<n<<" byte from server queue");
      if(n<=0)  break;
    }
    SetTimeout();
  }

  inline ssize_t ReceiveFromServer (void *buffer, int buffer_length, bool use_error_handler=true)
  {
    struct sockaddr_in  from_addr;
    socklen_t  fa_len= sizeof(sockaddr_in);
    ssize_t n= recvfrom(SockFD, buffer, buffer_length, 0, (struct sockaddr *)&from_addr, &fa_len);
    //!\todo <b>FIXME: error check for from_addr, fa_len </b>
    //!\todo <b>FIXME: use TIMEOUT </b>
    if (n<=0 || fa_len<=0)
    {
      perror("recvfrom"); LDBGVAR(n); LDBGVAR(fa_len);
      if (use_error_handler && ConnectionErrorHandler)
        ConnectionErrorHandler();
    }
    return n;
  }

  template <typename t_data>
  inline ssize_t ReceiveFromServer (t_data &v_data, bool use_error_handler=true)
  {
    struct sockaddr_in  from_addr;
    socklen_t  fa_len= sizeof(sockaddr_in);
    ssize_t n= recvfrom(SockFD, &v_data, sizeof(t_data), 0, (struct sockaddr *)&from_addr, &fa_len);
    //!\todo <b>FIXME: error check for from_addr, fa_len </b>
    //!\todo <b>FIXME: use TIMEOUT </b>
    if (n<=0 || fa_len<=0)
    {
      perror("recvfrom"); LDBGVAR(n); LDBGVAR(fa_len);
      if (use_error_handler && ConnectionErrorHandler)
        ConnectionErrorHandler();
    }
    return n;
  }

  template <typename t_data>
  inline ssize_t SendToServer (const t_data &v_data)
  {
    return sendto (SockFD, &v_data, sizeof(t_data), 0, (struct sockaddr *)&ServerAddress, sizeof(ServerAddress));
  }

  inline ssize_t SendToServer (const char *v_data, int v_length)
  {
    return sendto (SockFD, v_data, v_length, 0, (struct sockaddr *)&ServerAddress, sizeof(ServerAddress));
  }
}

#include <string>
#include <sstream>
using namespace client;
using namespace std;

int main(int argc,char **argv)
{
  string address("127.0.0.1");
  int port(9877);
  if(argc>=2)  address= argv[1];
  if(argc>=3)
  {
    stringstream ss(argv[2]);
    ss>>port;
  }
  Setup (address.c_str(), port);

  SendToServer("init",4);

  char data[256];
  while(true)
  {
    int len=ReceiveFromServer(data,sizeof(data)-1);
    if(len>0)
    {
      data[len]='\0';
      cout<<data<<endl;
      if(data==string("quit"))  break;
    }
  }
  return 0;
}

