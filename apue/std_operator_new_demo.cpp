#include <iostream>
#include <new>


class ZeroedObject
{
public:
	void* operator new(size_t size);
	void operator delete(void* p);
	void* operator new[](size_t size);
	void operator delete[](void* p);
};

void* ZeroedObject::operator new(size_t size)
{
	std::cout << "operator new\n";
	void* ptr = NULL;
	if (size != 0)
	{
		ptr = ::operator new(size);
		if(ptr == NULL)
			throw std::bad_alloc();
		// throw bad_alloc if failed
	}
	return ptr;
}

void ZeroedObject::operator delete(void* p)
{
	std::cout << "operator delete\n";
	::operator delete(p);
}

void* ZeroedObject::operator new[](size_t size)
{
	std::cout << "operator new[]\n";
	void* ptr = NULL;
	if (size != 0)
	{
		ptr = ::operator new[](size);
		if(ptr == NULL)
			throw std::bad_alloc();
		// throw bad_alloc if failed
	}
	return ptr;
}

void ZeroedObject::operator delete[](void* p)
{
	std::cout << "operator delete[]\n";
	::operator delete[](p);
}

int main()
{
	ZeroedObject* pp1 = new ZeroedObject[4];
	delete[] pp1;

	ZeroedObject* pp = new ZeroedObject;
	delete pp;
}

