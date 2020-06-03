// See README.txt for information and build instructions.

#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>

#include "addressbook.pb.h"

using namespace std;
using google::protobuf::util::TimeUtil;

// Iterates though all people in the AddressBook and prints info about them.
void ListPeople(const tutorial::AddressBook& address_book) 
{
    for (int i = 0; i < address_book.people_size(); i++) 
    {
        const tutorial::Person& person = address_book.people(i);
        cout << "Person ID: " << person.id() << endl;
        cout << "  Name: " << person.name() << endl;
        if (person.email() != "") {
          cout << "  E-mail address: " << person.email() << endl;
        }

        for (int j = 0; j < person.phones_size(); j++) 
        {
            const tutorial::Person::PhoneNumber& phone_number = person.phones(j);
            switch (phone_number.type()) 
            {
                case tutorial::Person::MOBILE:
                    cout << "  Mobile phone #: ";
                    break;
                case tutorial::Person::HOME:
                    cout << "  Home phone #: ";
                    break;
                case tutorial::Person::WORK:
                    cout << "  Work phone #: ";
                    break;
                default:
                    cout << "  Unknown phone #: ";
                    break;
            }
            cout << phone_number.number() << endl;
        }

        const auto& nation = person.nation();
        cout << "  Country: " << nation.country() << endl;
        cout << "  Province: " << nation.province() << endl;
        cout << "  City: " << nation.city() << endl;

        if (person.has_last_updated()) 
        {
            cout << "  Updated: " << TimeUtil::ToString(person.last_updated()) << endl;
        }
    }
}

int main(int argc, char* argv[])
{
    // Verify that the library's version we linked against
    // is compatible with the headers' version we compiled against
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if(argc != 2)
    {
        cerr << "Usage: " << argv[0] << " Address_book_file" << endl;
        return -1;
    }

    tutorial::AddressBook book;
    fstream input(argv[1], ios::in|ios::binary);
    if(!input)
    {
        cerr << "Failed to open " << argv[1] << ", error: " << strerror(errno) << endl;
        return -1;
    }
    else if(!book.ParseFromIstream(&input))
    {
        cerr << "Failed to parse address book" << endl;
        return -1;
    }

    //cout << "SpaceUsedLong(): " << book.SpaceUsedLong() << endl;
    cout << "ByteSizeLong(): " << book.ByteSizeLong() << endl;

    ListPeople(book);

    // optional: Delete all global objects allocatied by libprotobuf
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
