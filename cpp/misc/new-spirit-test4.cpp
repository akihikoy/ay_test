#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <list>
#include <boost/spirit/include/qi.hpp>
// #include <boost/spirit/include/phoenix.hpp>
using namespace boost;
using namespace boost::spirit;

namespace calc_action
{

std::list<long double> Stack;

void Push(const long double &val)
{
  Stack.push_back(val);
}
void PushS(const std::string &val)
{
  using namespace std;
  cout<<"push "<<val<<endl;
  Stack.push_back(atof(val.c_str()));
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
  typedef qi::rule<t_iterator, std::string(), standard::space_type> rule_sv;
  typedef qi::rule<t_iterator, char(), standard::space_type> rule_cv;
  typedef qi::rule<t_iterator, void(), standard::space_type> rule_vv;
  rule_sv statement, statement_unexpected;
  rule_sv literal_real;
  rule_vv expr, term, fctr;

  static void SyntaxError(std::string &val)
    {
      std::cout<<"syntax error!"<<std::endl;
      std::cout<<"  > "<<val<<std::endl;
    }

  calc() // : calc::base_type(expr)
  {
    using namespace boost::spirit::standard;
    statement
      = (
        expr
        | statement_unexpected [&SyntaxError]
        );
    statement_unexpected
      = (char_ - eol) >> *(char_ - eol);
    expr = term
        >> *( ('+' >> term [&calc_action::Add])
            | ('-' >> term [&calc_action::Subt]) );
    term = fctr
        >> *( ('*' >> fctr [&calc_action::Mult])
            | ('/' >> fctr [&calc_action::Div]) );
    // fctr = long_double [&calc_action::Push] | '(' >> expr >> ')';
    // fctr = qi::real_parser<long double, qi::strict_real_policies<long double> >() [&calc_action::Push] | '(' >> expr >> ')';
    fctr = literal_real [&calc_action::PushS] | '(' >> expr >> ')';

    rule_cv tmp_sign= char_('+') | char_('-');
    literal_real
      = -(string("+") | string("-"))
        >>(
            (
              +digit >> char_('.') >> *digit
                >> !( (string("e") | string("E")) >> -(string("+") | string("-")) >> +digit )
            )
          | string("inf")
          );
  }
};

int main(int argc,char**argv)
{
  using namespace std;
  string str=(argc>1)?argv[1]:"1*2";

  calc_action::Push(-1); // dummy

  calc<string::iterator> c;

  string::iterator it = str.begin();
  if( qi::phrase_parse(it, str.end(), c.statement, standard::space) )
  {
    cout << ((it==str.end())?"whole text has been parsed":"some syntax error") <<endl;
    cout << str << "= " << calc_action::Pop() << endl;
  }
}
