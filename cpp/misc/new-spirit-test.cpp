// http://www.kmonos.net/alang/boost/classes/spirit.html ���饳�ԡ�.
// �Ǥ⥳��ѥ���Ǥ��ʤ�
// boost �ΥС�������㤤��ǽ�����⤤
// -->NOTE: libboost1.42-dev ���ȥ���ѥ���Ǥ���

#include <iostream>
#include <string>
#include <vector>
#include <boost/fusion/tuple.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
using namespace std;
using namespace boost;
using namespace boost::spirit;

//
// ��1: ������ƥ�� int_ �򥫥�� ',' �Ƕ��ڤä��ꥹ�� (% �϶��ڤ�ʸ������Υꥹ�Ȳ���)
//
bool parse_csv( const string& str, std::vector<int>& result )
{
	string::const_iterator it = str.begin();
	return qi::parse(it, str.end(), int_ % ',', result) && it==str.end();
}

//
// ��2: �¿��Υڥ����Υꥹ�ȡ�>> ��ñ��ʸ������¤ӡ�+ ��ñ�ʤ룱�İʾ�η����֤���
//
bool parse_points( const string& str, std::vector< fusion::tuple<double,double> >& result )
{
	string::const_iterator it = str.begin();
	return qi::phrase_parse(it, str.end(),
		+( "x:" >> double_ >> "y:" >> double_ ),
		ascii::space, // ����˶��������Ƥ�̵�뤹��
		result
	) && it==str.end();
}

//
// ��3: ���������Ȥ����㡣������׻����ޤ�
//	 ʸˡ���
//	   expr ::= term ('+' term | '-' term)*
//	   term ::= fctr ('*' fctr | '/' fctr)*
//	   fctr ::= int | '(' expr ')'
//	 ���ϵ���
//	   expr
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
int main()
{
	{
		std::vector<int> result;
		if( parse_csv( "1,2,3", result ) )
			cout << result.size() << " integers parsed" << endl;
		else
			cout << "error" << endl;
	}
	{
		std::vector<int> result;
		if( parse_csv( "1 2 3", result ) ) // ����޶��ڤ�Ǥʤ�
			cout << result.size() << " integers parsed" << endl;
		else
			cout << "error" << endl;
	}
	{
		std::vector< fusion::tuple<double,double> > result;
		if( parse_points( "x: 1 y: 2.0 x: 3 y: -4.5", result ) )
			cout << result.size() << " pts parsed" << endl;
		else
			cout << "error" << endl;
	}
	for(string str; getline(cin,str) && str.size()>0; )
	{
		calc<string::iterator> c;

		int result = -1;

		string::iterator it = str.begin();
		if( qi::phrase_parse(it, str.end(), c, ascii::space, result) )
			cout << result << endl;
	}
}
