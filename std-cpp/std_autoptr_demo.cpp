#include <iostream>
#include <memory>

using namespace std;

struct Point3d {
	int x, y, z;
};

ostream& operator<<(ostream& out, const Point3d& p) {
	out << '(' << p.x << ',' << p.y << ',' << p.z << ')';
	return out;
}

int main() {
	Point3d* p = new Point3d;
	auto_ptr<Point3d> ap(p);
	ap->x = 1;
	ap->y = 2;
	ap->z = 3;
	cout << *ap << '\n';

	auto_ptr<Point3d> p2(ap);
	if (ap.get() == nullptr) {
		cout << "ap is null after {auto_ptr<Point3d> p2(ap);}" << endl;
	}
	cout << *p2 << endl;
	return 0;
}

