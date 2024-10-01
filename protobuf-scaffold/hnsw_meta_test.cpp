
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/text_format.h>
#include "hnsw_meta.pb.h"

using namespace std;

using HnswMeta = algorithm::hnsw::HnswMeta;

int load_from_file(const std::string& file_path, HnswMeta& meta) {
  std::filesystem::path meta_file_path = file_path;
  if (!std::filesystem::exists(meta_file_path)) {
    printf("%s does not exist\n", file_path.c_str());
    return 1;
  }
  std::ifstream input_file_pbtxt(file_path);
  if (!input_file_pbtxt.is_open()) {
    printf("load_from_file: %s input_file is not open, error: %s\n",
                file_path.c_str(), std::strerror(errno));
    return 2;
  }
  std::stringstream input_buffer;
  input_buffer << input_file_pbtxt.rdbuf();
  std::string input_string = input_buffer.str();

  if (file_path.find(".pbjson") != std::string::npos) {
    google::protobuf::util::JsonParseOptions jparseopt;
    jparseopt.ignore_unknown_fields = true;
    auto status = google::protobuf::util::JsonStringToMessage(input_string, &meta, jparseopt);
    if (!status.ok()) {
      printf("load_from_file input string from path: %s has wrong proto\n",
                  file_path.c_str());
      return 3;
    }
  } else if (file_path.find(".pbtxt") != std::string::npos) {
      google::protobuf::TextFormat::Parser parser;
      parser.AllowUnknownField(true);
      bool status = parser.ParseFromString(input_string, &meta);
      if (status == false) {
        printf("load_from_file: %s has wrong proto\n",
                    file_path.c_str());
        return 4;
      }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s meta.pbjson/meta.pbtxt\n", argv[0]);
    return -1;
  }
  HnswMeta meta_info;
  int ret = load_from_file(argv[1], meta_info);
  if (ret != 0) {
    printf("failed to load_from_file, ret=%d\n", ret);
    return ret;
  }

  //cout << "meta_info.DebugString: " << meta_info.DebugString() << endl;
  cout << "meta_info.model_info: " << meta_info.model_info() << endl;
  cout << "meta_info.element_count: " << meta_info.element_count() << endl;
  
}

/*
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/hnsw_meta.proto
# change to the actual path of protobuf lib
g++ -I. -I/opt/homebrew/Cellar/protobuf/21.12/include -L/opt/homebrew/Cellar/protobuf/21.12/lib -lprotobuf hnsw_meta.pb.cc hnsw_meta_test.cpp -o hnsw_meta_test -std=c++17
*/

/* test json
{
 "label_type": "LB_UINT64",
 "vec_type": "VEC_FLOAT32",
 "vec_dim": 128,
 "m": 32,
 "cef": 600,
 "sef": 600,
 "element_count": 1234,
 "model_info": "verbose info",
}
*/