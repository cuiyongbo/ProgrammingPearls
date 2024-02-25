// See README.txt for information and build instructions.

#include <ctime>
#include <fstream>
#include <google/protobuf/util/time_util.h>
#include <iostream>
#include <string>

#include "addressbook.pb.h"

using namespace std;

using google::protobuf::util::TimeUtil;

// This function fills in a Person message based on user input.
void PromptForAddress(tutorial::Person* person) {
    cout << "Enter person ID number: ";
    int id;
    cin >> id;
    person->set_id(id);
    cin.ignore(256, '\n');

    cout << "Enter name: ";
    getline(cin, *person->mutable_name());

    cout << "Enter email address (blank for none): ";
    string email;
    getline(cin, email);
    if (!email.empty()) {
      person->set_email(email);
    }

    while (true) {
        cout << "Enter a phone number (or leave blank to finish): ";
        string number;
        getline(cin, number);
        if (number.empty()) {
          break;
        }

        tutorial::Person::PhoneNumber* phone_number = person->add_phones();
        phone_number->set_number(number);

        cout << "Is this a mobile, home, or work phone? ";
        string type;
        getline(cin, type);
        if (type == "mobile") {
          phone_number->set_type(tutorial::Person::MOBILE);
        } else if (type == "home") {
          phone_number->set_type(tutorial::Person::HOME);
        } else if (type == "work") {
          phone_number->set_type(tutorial::Person::WORK);
        } else {
          cout << "Unknown phone type.  Using default." << endl;
        }
    }
    
    auto nation = person->mutable_nation();
    string ss;
    cout << "Enter your country: ";
    getline(cin, ss);
    nation->set_country(ss);
    cout << "Enter your province: ";
    getline(cin, ss);
    nation->set_province(ss);
    cout << "Enter your city: ";
    getline(cin, ss);
    nation->set_city(ss);

    *person->mutable_last_updated() = TimeUtil::SecondsToTimestamp(time(NULL));
}

int main(int argc, char* argv[]) {
    // Verify that the library's version we linked against
    // is compatible with the headers' version we compiled against
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " Address_book_file" << endl;
        return -1;
    }

    tutorial::AddressBook book;
    fstream input(argv[1], ios::in|ios::binary);
    if(!input) {
        cout << argv[1] << ": File not found. Creating a new file." << endl;
    } else if(!book.ParseFromIstream(&input)) {
        cerr << "Failed to parse address book" << endl;
        return -1;
    }

    PromptForAddress(book.add_people());

    fstream output(argv[1], ios::out|ios::trunc|ios::binary);
    if(!book.SerializeToOstream(&output)) {
        cerr << "Failed to write address book" << endl;
        return -1;
    }

    // optional: Delete all global objects allocatied by libprotobuf
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
