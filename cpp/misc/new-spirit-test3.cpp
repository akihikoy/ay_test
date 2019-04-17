#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <list>
#include <boost/spirit/include/qi.hpp>
// #include <boost/spirit/include/phoenix.hpp>
using namespace std;
using namespace boost;
using namespace boost::spirit;

namespace calc_action
{

std::list<long double> Stack;

void Push(const long double &val)
{
  Stack.push_back(val);
}
long double Pop()
{
  long double res= Stack.back();
  Stack.pop_back();
  return res;
}

void Add()
{
  long double rhs(Pop()),lhs(Pop());
  Push(lhs+rhs);
}
void Subt()
{
  long double rhs(Pop()),lhs(Pop());
  Push(lhs-rhs);
}
void Mult()
{
  long double rhs(Pop()),lhs(Pop());
  Push(lhs*rhs);
}
void Div()
{
  long double rhs(Pop()),lhs(Pop());
  Push(lhs/rhs);
}

}

template<typename t_iterator>
struct calc
  // : qi::grammar<t_iterator>
{
  qi::rule<t_iterator, void(), ascii::space_type> expr, term, fctr;

  calc() // : calc::base_type(expr)
  {
    expr = term
        >> *( ('+' >> term [&calc_action::Add])
            | ('-' >> term [&calc_action::Subt]) );
    term = fctr
        >> *( ('*' >> fctr [&calc_action::Mult])
            | ('/' >> fctr [&calc_action::Div]) );
    fctr = long_double [&calc_action::Push] | '(' >> expr >> ')';
  }
};

int main(int argc,char**argv)
{
  string str=(argc>1)?argv[1]:"1*2";

  calc<string::iterator> c;

  string::iterator it = str.begin();
  if( qi::phrase_parse(it, str.end(), c.expr, ascii::space) )
    cout << str << "= " << calc_action::Pop() << endl;
}
