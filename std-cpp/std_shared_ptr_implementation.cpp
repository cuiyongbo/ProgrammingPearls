#include <iostream>

using std::cout;
using std::endl;

/*
      __shared_count(const __shared_count& __r) noexcept
      : _M_pi(__r._M_pi)
      {
	if (_M_pi != 0)
	  _M_pi->_M_add_ref_copy();
      }

      __shared_count&
      operator=(const __shared_count& __r) noexcept
      {
	_Sp_counted_base<_Lp>* __tmp = __r._M_pi;
	if (__tmp != _M_pi)
	  {
	    if (__tmp != 0)
	      __tmp->_M_add_ref_copy();
	    if (_M_pi != 0)
	      _M_pi->_M_release();
	    _M_pi = __tmp;
	  }
	return *this;
      }

      void
      _M_swap(__shared_count& __r) noexcept
      {
	_Sp_counted_base<_Lp>* __tmp = __r._M_pi;
	__r._M_pi = _M_pi;
	_M_pi = __tmp;
      }
*/

/*
key points:
    1. separate reference_count object from resource to be managed
    2. acquire ownership of new resource, you need initialize the ownership
    3. to acquire ownership from another shared_ptr, first release current ownership, then update reference_count
    4. after releasing ownership you need release the resource if reference_count reaches 0
    5. the reference_count `m_count` has to be of pointer type because the reference changes must be 
        transported to the original instance which can't be achieved using non-pointer type
*/

class shared_ptr_count {
public:
    shared_ptr_count():
        m_count(nullptr) {
    }

    shared_ptr_count(const shared_ptr_count& rhs):
        m_count(rhs.m_count) {
    }

    void swap(shared_ptr_count& rhs) {
        std::swap(m_count, rhs.m_count);
    }

    long use_count () const {
        long count = 0;
        if (m_count != nullptr) {
            count = *m_count;
        }
        return count;
    }

    template<typename T>
    void acquire(T* p) {
        if (p != nullptr) {
            if (m_count != nullptr) {
                ++(*m_count);
            } else {
                m_count = new long(1);
            }
        }
    }

    template<typename T>
    void release(T* p) {
        if (p != nullptr) {
            --(*m_count);
            if (*m_count == 0) {
                delete p;
                delete m_count;
            }
            m_count = nullptr;
        }
    }
private:
    long* m_count;
};

template <typename T> 
class naive_shared_ptr {
public:
    naive_shared_ptr():
        m_ptr(nullptr),
        m_ref_count(nullptr) {
    }

    naive_shared_ptr(T* ptr):
        naive_shared_ptr() {
        if (ptr != nullptr) {
            m_ptr = ptr;
            m_ref_count = new int64_t(1);
        } 
    }

    ~naive_shared_ptr() {
        reset();
    }

    T* get() {
        return m_ptr;
    }

    T* operator->() {
        return m_ptr;
    }

    T& operator*() {
        return *m_ptr;
    }

    int64_t use_count() {
        int64_t ref = 0;
        if (m_ref_count != nullptr) {
            ref = *m_ref_count; 
        }
        return ref;
    }

    void reset() {
        // Releases the ownership of the managed object, if any. After the call, *this manages no object. 
        if (m_ref_count != nullptr) {
            *m_ref_count = *m_ref_count - 1;
            if (*m_ref_count == 0) {
                delete m_ref_count;
                delete m_ptr;
            }
            m_ptr = nullptr;
            m_ref_count = nullptr;
        }
    }

    void reset(T* rhs) {
        if (rhs != m_ptr) {
            reset();
            m_ref_count = new int64_t(1);
            m_ptr = rhs;
        }
    }

    naive_shared_ptr(const naive_shared_ptr& rhs):
        m_ptr(rhs.m_ptr),
        m_ref_count(rhs.m_ref_count) {
        *m_ref_count = *m_ref_count + 1;
    }

    naive_shared_ptr* operator=(const naive_shared_ptr& rhs) {
        if (m_ptr != rhs.m_ptr) {
            reset();
            m_ref_count = rhs.m_ref_count;
            *m_ref_count = *m_ref_count + 1;
            m_ptr = rhs.m_ptr;
        }
        return *this;
    }

    void swap(naive_shared_ptr& rhs) {
        std::swap(m_ptr, rhs.m_ptr);
        std::swap(m_ref_count, rhs.m_ref_count);
    }

private:
    T* m_ptr;
    int64_t* m_ref_count;
};

template<typename T> 
class shared_ptr {
public:
    shared_ptr():
        m_pointer(nullptr) {
    }

    explicit shared_ptr(T* p) {
        acquire(p);
    }

    shared_ptr(const shared_ptr& rhs):
        m_ref_count(rhs.m_ref_count) {
        acquire(rhs.m_pointer);
    }

    shared_ptr& operator=(const shared_ptr& rhs) {
        // release ownership of this->m_pointer
        release();

        // copy rhs.m_ref_count
        m_ref_count = rhs.m_ref_count;

        // shared ownership of rhs.m_pointer
        acquire(rhs.m_pointer);
    }

    void swap(shared_ptr& rhs) {
        m_ref_count.swap(rhs.m_ref_count);
        std::swap(m_pointer, rhs.m_pointer);
    }

    ~shared_ptr() {
        release();
    }

    void reset() {
        release();
    }

    void reset(T* p) {
        release();
        acquire(p);
    }

    long use_count() const {
        return m_ref_count.use_count();
    }

    inline T* get() {
        return m_pointer;
    }

    inline T* operator->() {
        return m_pointer;
    }

    inline T& operator*() {
        return *m_pointer;
    }

private:
    // acquire/share the ownership of pointer p
    void acquire(T* p) {
        m_ref_count.acquire(p);
        m_pointer = p;
    }

    // release ownership if reference reaches 0
    // this function will reset managed resource
    void release() {
        m_ref_count.release(m_pointer);
        m_pointer = nullptr;
    }

private:
    T* m_pointer;
    shared_ptr_count m_ref_count;
};

void test01() {
class Foo {
public:
    Foo() {
        cout << "Foo()" << endl;
    }
    ~Foo() {
        cout << "~Foo()" << endl;
    }    
};

    {
        cout << "null shared_ptr" << endl;
        shared_ptr<Foo> o1;
    }

    {
        cout << "constuctor with object" << endl;
        shared_ptr<Foo> o1(new Foo);
        shared_ptr<Foo> o2(o1);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;

        cout << "after o2.reset" << endl;
        Foo* p = new Foo;
        o2.reset(p);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;

        cout << "after shared_ptr<Foo> o3(o1)" << endl;
        shared_ptr<Foo> o3(o1);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;        
        cout << "o3: " << o3.use_count() << endl;        

        //swap(o2, o1);
        o2.swap(o1);
        cout << "after swap(o2, o1)" << endl;
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;        
        cout << "o3: " << o3.use_count() << endl;         
    }
}


void test02() {
class Foo {
public:
    Foo() {
        cout << "Foo()" << endl;
    }
    ~Foo() {
        cout << "~Foo()" << endl;
    }    
};

    {
        cout << "null shared_ptr" << endl;
        naive_shared_ptr<Foo> o1;
    }

    {
        cout << "constuctor with object" << endl;
        naive_shared_ptr<Foo> o1(new Foo);
        naive_shared_ptr<Foo> o2(o1);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;

        cout << "after o2.reset" << endl;
        Foo* p = new Foo;
        o2.reset(p);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;

        cout << "after shared_ptr<Foo> o3(o1)" << endl;
        naive_shared_ptr<Foo> o3(o1);
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;        
        cout << "o3: " << o3.use_count() << endl;        

        //swap(o2, o1);
        o2.swap(o1);
        cout << "after swap(o2, o1)" << endl;
        cout << "o1: " << o1.use_count() << endl;
        cout << "o2: " << o2.use_count() << endl;        
        cout << "o3: " << o3.use_count() << endl;         
    }
}


int main() {
    test01();
    test02();
}

