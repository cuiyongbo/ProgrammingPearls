#include <string>
#include <fstream>
#include <iostream>

#include <google/protobuf/util/time_util.h>
#include <google/protobuf/message.h>

#include "addressbook.pb.h"

using namespace std;
using google::protobuf::util::TimeUtil;
using Message = google::protobuf::Message;
using Descriptor = google::protobuf::Descriptor;
using Reflection = google::protobuf::Reflection;
using FieldDescriptor = google::protobuf::FieldDescriptor;


// Iterates though all people in the AddressBook and prints info about them.
void ListPeople(const tutorial::AddressBook& address_book) {
    for (int i = 0; i < address_book.people_size(); i++) {
        const tutorial::Person& person = address_book.people(i);
        cout << "Person ID: " << person.id() << endl;
        cout << "  Name: " << person.name() << endl;
        if (person.email() != "") {
          cout << "  E-mail address: " << person.email() << endl;
        }
        for (int j = 0; j < person.phones_size(); j++) {
            const tutorial::Person::PhoneNumber& phone_number = person.phones(j);
            switch (phone_number.type()) {
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
        if (person.has_last_updated()) {
            cout << "  Updated: " << TimeUtil::ToString(person.last_updated()) << endl;
        }
    }
}


// full_field_name may be a nested field, e.g. foo.bar
void print_schema(const Message& msg, const std::string& sub_field_name, const std::string& full_field_name) {
    const Descriptor* descriptor = msg.GetDescriptor();
    const Reflection* reflection = msg.GetReflection();
    auto print_field_detail = [&] (const Message& msg, const FieldDescriptor* pfield, const Reflection* reflection) {
        auto get_label_name = [&] () {
            if (pfield->is_required()) {
                return "required";
            } else if (pfield->is_optional()) {
                return "optional";
            } else if (pfield->is_repeated()) {
                return "repeated";
            } else {
                return "singular";
            }
        };
        cout << "field: " << pfield->full_name();
        cout << ", field index: " << pfield->number();
        cout << ", type: " << pfield->type_name();
        cout << ", label: " << get_label_name();
        if (pfield->is_repeated()) {
        } else {
            cout << ", value: ";
            if (pfield->type() == FieldDescriptor::TYPE_STRING) {
                cout << reflection->GetString(msg, pfield);
            } else if (pfield->type() == FieldDescriptor::TYPE_INT32) {
                cout << reflection->GetInt32(msg, pfield);
            } else if (pfield->type() == FieldDescriptor::TYPE_FLOAT) {
                cout << reflection->GetFloat(msg, pfield);
            } else if (pfield->type() == FieldDescriptor::TYPE_DOUBLE) {
                cout << reflection->GetDouble(msg, pfield);
            } else if (pfield->type() == FieldDescriptor::TYPE_BOOL) {
                cout << reflection->GetBool(msg, pfield);
            } else if (pfield->type() == FieldDescriptor::TYPE_ENUM) {
                cout << reflection->GetEnum(msg, pfield)->name();
            }
        }
        cout << endl;
    };

    int p = sub_field_name.find(".");
    if (p != std::string::npos) {
        auto field_name = sub_field_name.substr(0, p);
        auto next_field = sub_field_name.substr(p+1);
        const FieldDescriptor* pfield = descriptor->FindFieldByName(field_name);
        if (!pfield) {
            cout << "unknow field: " << field_name << ", full_field_name: " << full_field_name << endl;
            return;
        }
        print_field_detail(msg, pfield, reflection);
        if (pfield->is_repeated()) {
            const Message& sub_proto = reflection->GetRepeatedMessage(msg, pfield, 0);
            print_schema(sub_proto, next_field, full_field_name);
        } else {
            const Message& sub_proto = reflection->GetMessage(msg, pfield);
            print_schema(sub_proto, next_field, full_field_name);
        }
    } else {
        const FieldDescriptor* pfield = descriptor->FindFieldByName(sub_field_name);
        if (!pfield) {
            cout << "unknow field: " << sub_field_name << ", full_field_name: " << full_field_name << endl;
            return;
        }
        print_field_detail(msg, pfield, reflection);
    }
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
        cerr << "Failed to open " << argv[1] << ", error: " << strerror(errno) << endl;
        return -1;
    } else if(!book.ParseFromIstream(&input)) {
        cerr << "Failed to parse address book" << endl;
        return -1;
    }

    //cout << "SpaceUsed(): " << book.SpaceUsed() << endl;
    cout << "SpaceUsedLong(): " << book.SpaceUsedLong() << endl;
    cout << "ByteSizeLong(): " << book.ByteSizeLong() << endl;
    //cout << "DebugString(): " << book.DebugString() << endl;
    //print_schema(book, "people", "people");
    print_schema(book, "people.nation.country", "people.nation.country");
    ListPeople(book);

    // optional: Delete all global objects allocatied by libprotobuf
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
