// http://www.kmonos.net/alang/boost/classes/spirit.html からコピー.
// でもコンパイルできない
// boost のバージョンが低い可能性が高い
// -->NOTE: libboost1.42-dev だとコンパイルできた

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
// 例1: 整数リテラル int_ をカンマ ',' で区切ったリスト (% は区切り文字指定のリスト解析)
//
bool parse_csv( const string& str, std::vector<int>& result )
{
	string::const_iterator it = str.begin();
	return qi::parse(it, str.end(), int_ % ',', result) && it==str.end();
}

//
// 例2: 実数のペア、のリスト（>> は単に文字列の並び、+ は単なる１個以上の繰り返し）
//
bool parse_points( const string& str, std::vector< fusion::tuple<double,double> >& result )
{
	string::const_iterator it = str.begin();
	return qi::phrase_parse(it, str.end(),
		+( "x:" >> double_ >> "y:" >> double_ ),
		ascii::space, // 途中に空白を入れても無視する
		result
	) && it==str.end();
}

//
// 例3: 少しちゃんとした例。数式を計算します
//	 文法定義
//	   expr ::= term ('+' term | '-' term)*
//	   term ::= fctr ('*' fctr | '/' fctr)*
//	   fctr ::= int | '(' expr ')'
//	 開始記号
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

	// 計算ついでに、* と / を見ると意味もなく Oops! と叫ぶコード
	static void Oops() { cout << "Oops!" << endl; }
};

// 使用例
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
		if( parse_csv( "1 2 3", result ) ) // カンマ区切りでない
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
