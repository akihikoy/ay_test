// http://www.kmonos.net/alang/boost/classes/spirit.html ���饳�ԡ�.

#include <iostream>
#include <string>
#include <vector>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
using namespace std;
using namespace boost;
using namespace boost::spirit;

//
// ��3: ���������Ȥ����㡣������׻����ޤ�
//   ʸˡ���
//     expr ::= term ('+' term | '-' term)*
//     term ::= fctr ('*' fctr | '/' fctr)*
//     fctr ::= int | '(' expr ')'
//   ���ϵ���
//     expr
//
template<typename Iterator>
struct calc
  : qi::grammar<Iterator, int(), ascii::space_type>
{
  qi::rule<Iterator, int(), ascii::space_type> expr, term, fctr;

  calc() : calc::base_type(expr)
  {
    expr = term[_val = _1] >> *( ('+' >> term[_val += _1])
                               | ('-' >> term[_val -= _1]) );
    term = fctr[_val = _1] >> *( ('*' >> fctr[_val *= _1])
                               | ('/' >> fctr[_val /= _1]) )[&Oops];
    fctr = int_ | '(' >> expr >> ')';
  }

  // �׻��Ĥ��Ǥˡ�* �� / �򸫤�Ȱ�̣��ʤ� Oops! �ȶ��֥�����
  static void Oops() { cout << "Oops!" << endl; }
};

// ������
int main(int argc,char**argv)
{
  string str=(argc>1)?argv[1]:"1*2";

  calc<string::iterator> c;

  int result = -1;

  string::iterator it = str.begin();
  if( qi::phrase_parse(it, str.end(), c, ascii::space, result) )
    cout << str << "= " << result << endl;
}
