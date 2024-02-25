#include <cassert>
#include <iostream>
using std::ostream;

// c style
namespace s1
{
	typedef struct Point3D
	{
		float x, y, z;
	} Point3D;

#define Point3D_print(pd) printf("(%g, %g, %g)", pd->x, pd->y, pd->z)
}

// c++ style 1
namespace s2
{
	class Point3D
	{
	public:
		Point3D(float x=0.0, float y=0.0, float z=0.0)
			:m_x(x), m_y(y), m_z(z) {}

		float x() const { return m_x; }
		float y() const { return m_y; }
		float z() const { return m_z; }
		void x(float x) { m_x = x; }
		void y(float y) { m_y = y; }
		void z(float z) { m_z = z; }

	private:
		float m_x, m_y, m_z;
	};

	inline ostream& operator<< (ostream& os, const Point3D& pt)
	{
		os << "(" << pt.x() << "," << pt.y() << "," << pt.z() << ")";
	}
}

// c++ style 2
namespace s3
{
	class Point
	{
	public:
		Point(float x) : m_x(x) {}
		float x() const { return m_x; }
		void x(float x) { m_x = x; }
	protected:
		float m_x;
	};

	class Point2D : public Point
	{
	public:
		Point2D(float x, float y) : Point(x), m_y(y) {}
		float y() const { return m_y; }
		void y(float y) { m_y = y; }
	protected:
		float m_y;
	};

	class Point3D :public Point2D
	{
	public:
		Point3D(float x = 0.0, float y = 0.0, float z = 0.0)
			:Point2D(x, y), m_z(z) {}
		float z() const { return m_z; }
		void z(float z) { m_z = z; }
	private:
		float m_z;
	};
}

// c++ style 3
namespace s4
{
	template <class type, int dimension>
	class Point
	{
	public:
		Point(type coords[dimension])
		{
			for (int i = 0; i < dimension; i++)
				m_coordinates[i] = coords[i];
		}

		type& operator[](int index)
		{
			assert(index<dimension && index >=0);
			return m_coordinates[index];
		}

		type operator[](int index) const
		{
			assert(index<dimension && index >=0);
			return m_coordinates[index];
		}

	private:
		type m_coordinates[dimension];
	};

	template <class type, int dimension> inline
	ostream& operator<< (ostream& os, const Point<type, dimension>& pt)
	{
		os << "(";
		for (int i = 0; i < dimension - 1; i++)
			os << pt[i] << ",";
		os << pt[dimension - 1] << ")";
	}
}

int main()
{
	int a[] = {1,2,3};
	s4::Point<int, 3> p(a);
	p[2] = 5;
	std::cout << p << '\n';
}
