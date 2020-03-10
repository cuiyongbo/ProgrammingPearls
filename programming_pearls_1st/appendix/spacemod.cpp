#include <iostream>
#include <iomanip>

using std::cout;

typedef unsigned char uint8;

#define MEASURE(T, text)	{	\
	cout << std::setw(10) << text << '\t';	\
	cout << std::setw(2) << sizeof(T) << '\t';	\
	uint8* lastp = 0;	\
	for(int i=0; i<11; i++) {	\
		T* p = new T;	\
		uint8* thisp = (uint8*)p;	\
		if(lastp != 0)	\
			cout << ' ' << thisp-lastp;	\
		lastp = thisp;	\
	}	\
	cout << '\n';	\
}

struct structc {char c;};
struct structc1 {char c[1];};
struct structc12 {char c[12];};
struct structc13 {char c[13];};
struct structc28 {char c[28];};
struct structc29 {char c[29];};
struct structc44 {char c[44];};
struct structc45 {char c[45];};
struct structc60 {char c[60];};
struct structc61 {char c[61];};
struct structic {int i; char c;};
struct structip {int i; structip* p;};
struct structdc {double d; char c;};
struct structcd {char c;double d;};
struct structcdc {char c1; double d; char c2;};
struct structiii {int i1; int i2; int i3;};
struct structiic {int i1; int i2; char c;};


int main()
{
	cout << "Raw sizeof";
	cout << "\nsizeof(char)=" << sizeof(char);
	cout << " sizeof(short)=" << sizeof(short);
	cout << " sizeof(int)=" << sizeof(int);
	cout << "\nsizeof(float)=" << sizeof(float);
	cout << " sizeof(long)=" << sizeof(long);
	cout << " sizeof(struct*)=" << sizeof(structc*);
	cout << "\nsizeof(double)=" << sizeof(double);

	cout << "\n\nMEASURE macro\n";
	MEASURE(int, "int");
	MEASURE(structc, "structc");
	MEASURE(structc1, "structc1");
	MEASURE(structc12, "structc12");
	MEASURE(structc13, "structc13");
	MEASURE(structc28, "structc28");
	MEASURE(structc29, "structc29");
	MEASURE(structc44, "structc44");
	MEASURE(structc45, "structc45");
	MEASURE(structc60, "structc60");
	MEASURE(structc61, "structc61");
	MEASURE(structic, "structic");
	MEASURE(structip, "structip");
	MEASURE(structdc, "structdc");
	MEASURE(structcd, "structcd");
	MEASURE(structcdc, "structcdc");
	MEASURE(structiii, "structiii");
	MEASURE(structiic, "structiic");

	return 0;
}

