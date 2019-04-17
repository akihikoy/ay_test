#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/base/Version.h>
#include <iostream>
using namespace shogun;
using namespace std;
void print_message(FILE* target, const char* str)
{
  fputs(str,target);
}
int main()
{
  init_shogun(print_message);

  get_global_version()->print_version();

  exit_shogun();
  return 0;
}
