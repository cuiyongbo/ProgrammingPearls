// A minimal but useful C++ example showing how to load an mnist object
// recognition TensorFlow model, prepare input images for it, run them through the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"


static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename,
                             tensorflow::Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  std::string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<tensorflow::tstring>()() = tensorflow::tstring(data);
  return tensorflow::OkStatus();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
tensorflow::Status ReadTensorFromImageFile(const std::string& file_name, std::vector<tensorflow::Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();

  std::string input_name = "file_reader";
  std::string output_name = "normalized";

  // read file_name into a tensor named input
  tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader = tensorflow::ops::Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
    {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 1;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = tensorflow::ops::DecodePng(root.WithOpName("png_reader"), file_reader,
                             tensorflow::ops::DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        tensorflow::ops::Squeeze(root.WithOpName("squeeze_first_dim"),
                tensorflow::ops::DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = tensorflow::ops::DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              tensorflow::ops::DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = tensorflow::ops::Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = tensorflow::ops::ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = tensorflow::ops::ResizeBilinear(
      root, dims_expander,
      tensorflow::ops::Const(root.WithOpName("size"), {28, 28}));
  auto reshape = tensorflow::ops::Reshape(root, resized, {-1, 28*28});
  tensorflow::ops::Div(root.WithOpName(output_name), reshape, {(float)255.0});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return tensorflow::OkStatus();
}

tensorflow::Status GetPredictedLabel(const std::vector<tensorflow::Tensor>& outputs, int& predicted_label, bool verbose) {
  auto root = tensorflow::Scope::NewRootScope();
  std::string output_name = "arg_max";
  tensorflow::ops::ArgMax(root.WithOpName(output_name), outputs[0], 1);
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<tensorflow::Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name+":0"}, {}, &out_tensors));
  predicted_label = out_tensors[0].vec<tensorflow::int64>()(0);
  if (verbose) {
    for (const auto& record : out_tensors) {
      LOG(INFO) << record.DebugString();
      LOG(INFO) << "DataType: " << tensorflow::DataTypeString(record.dtype()) << ", NumElements: " << record.NumElements();
      int dims = record.dims();
      LOG(INFO) << "dimensions: " << dims;
      for (int i=0; i<dims; ++i) {
        LOG(INFO) << "dimensions[" << i << "]=" << record.dim_size(i);
      }
      LOG(INFO) << "tensor values: " << out_tensors[0].vec<tensorflow::int64>()(0);
    }
  }
  return tensorflow::OkStatus();
}

/* Graph structure:

# saved_model_cli show --dir data/mnist/1 --tag_set serve
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"

# saved_model_cli show --dir data/mnist/1 --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['dense_2_input'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 784)
      name: serving_default_dense_2_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['dense_3'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict

# run the binary with:  
# ./mnist_demo_main --image=data/mnist_test_demo.jpg --graph=data/mnist/1 --input_layer=serving_default_dense_2_input:0 --output_layer=StatefulPartitionedCall:0
*/

namespace tensorflow {
namespace cc_op {
// tensorflow/tensorflow/cc/ops/cc_op_gen_util.cc
template <typename T>
string PrintArray(int64_t num_elts, const T* array) {
  string ret;
  for (int64_t i = 0; i < num_elts; ++i) {
    if (i > 0) strings::StrAppend(&ret, ", ");
    strings::StrAppend(&ret, array[i]);
  }
  return ret;
}

std::string PrintTensorShape(const tensorflow::TensorShapeProto& shape_proto) {
  tensorflow::PartialTensorShape shape(shape_proto);
  if (shape.IsIdenticalTo(tensorflow::PartialTensorShape())) {
    return "::tensorflow::PartialTensorShape() /* unknown */";
  }
  std::string ret = "{";
  for (int d = 0; d < shape.dims(); ++d) {
    if (d > 0) tensorflow::strings::StrAppend(&ret, ", ");
    tensorflow::strings::StrAppend(&ret, shape.dim_size(d));
  }
  tensorflow::strings::StrAppend(&ret, "}");
  return ret;
}

string PrintTensor(const TensorProto& tensor_proto) {
  Tensor t(tensor_proto.dtype());
  CHECK(t.FromProto(tensor_proto));
  const int64_t num_elts = t.NumElements();
  switch (t.dtype()) {
    case DT_FLOAT:
      return PrintArray(num_elts, t.flat<float>().data());
    case DT_DOUBLE:
      return PrintArray(num_elts, t.flat<double>().data());
    case DT_INT32:
      return PrintArray(num_elts, t.flat<int32>().data());
    case DT_UINT8:
    case DT_QUINT8:
      return PrintArray(num_elts, t.flat<uint8>().data());
    case DT_UINT16:
    case DT_QUINT16:
      return PrintArray(num_elts, t.flat<uint16>().data());
    case DT_INT16:
    case DT_QINT16:
      return PrintArray(num_elts, t.flat<int16>().data());
    case DT_INT8:
    case DT_QINT8:
      return PrintArray(num_elts, t.flat<int8>().data());
    case DT_INT64:
      return PrintArray(num_elts, t.flat<int64_t>().data());
    case DT_BOOL:
      return PrintArray(num_elts, t.flat<bool>().data());
    case DT_STRING: {
      string ret;
      for (int64_t i = 0; i < num_elts; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        strings::StrAppend(&ret, str_util::CEscape(t.flat<tstring>()(i)));
      }
      return ret;
    }
    default: {
      LOG(FATAL) << "Not handling type " << DataType_Name(t.dtype());
      return string();
    }
  }
}

string PrintTensorProto(const TensorProto& proto) {
  return strings::StrCat("Input::Initializer(", "{", PrintTensor(proto), "}, ",
                         PrintTensorShape(proto.tensor_shape()),
                         ").AsTensorProto()");
}

} // cc_op
} // tensorflow


namespace tensorflow {

// tensorflow/tensorflow/cc/tools/freeze_saved_model.cc
// Gets tensor names from tensor_info and inserts them into the set of tensor names.
void GetTensorNamesFromTensorInfo(const TensorInfo& tensor_info,
                                  std::unordered_set<string>* tensor_names) {
  if (tensor_info.has_coo_sparse()) {
    // If the tensor is sparse we have to add all three tensors of the sparse
    // representations.
    const TensorInfo_CooSparse& coo_sparse = tensor_info.coo_sparse();
    tensor_names->insert(coo_sparse.values_tensor_name());
    tensor_names->insert(coo_sparse.indices_tensor_name());
    tensor_names->insert(coo_sparse.dense_shape_tensor_name());
  } else if (tensor_info.has_composite_tensor()) {
    for (const auto& component : tensor_info.composite_tensor().components()) {
      tensor_names->insert(component.name());
    }
  } else {
    tensor_names->insert(tensor_info.name());
  }
}

// Gets the union of all inputs and outputs of all SignatureDefs in the bundle
void GetSignatureDefsInputsAndOutputs(
    const SavedModelBundle& saved_model_bundle,
    std::unordered_set<string>* inputs, std::unordered_set<string>* outputs) {
  for (auto& sigdef_elem : saved_model_bundle.meta_graph_def.signature_def()) {
    const SignatureDef& signature_def = sigdef_elem.second;
    for (auto& input_elem : signature_def.inputs()) {
      GetTensorNamesFromTensorInfo(input_elem.second, inputs);
    }
    for (auto& output_elem : signature_def.outputs()) {
      GetTensorNamesFromTensorInfo(output_elem.second, outputs);
    }
  }
}

} // tensorflow


int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than mnist, then you'll need to update these.
  std::string image = "data/mnist_demo.jpg";
  std::string graph = "./data/mnist_cpu";
  std::string input_layer = "serving_default_dense_input";
  std::string output_layer = "StatefulPartitionedCall";
  std::string root_dir = "";
  bool verbose = false;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("image", &image, "image to be processed"),
      tensorflow::Flag("graph", &graph, "graph to be executed"),
      tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
      tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
      tensorflow::Flag("root_dir", &root_dir, "interpret image and graph file names relative to this directory"),
      tensorflow::Flag("verbose", &verbose, "print more details about inference"),
  };
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<tensorflow::Tensor> resized_tensors;
  std::string image_path = tensorflow::io::JoinPath(root_dir, image);
  tensorflow::Status read_tensor_status = ReadTensorFromImageFile(image_path, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const tensorflow::Tensor& resized_tensor = resized_tensors[0];

  tensorflow::SavedModelBundle model;
  tensorflow::RunOptions run_options;
  tensorflow::SessionOptions session_options;
  std::string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  tensorflow::Status load_status = tensorflow::LoadSavedModel(
    session_options, 
    run_options, 
    graph_path, 
    {tensorflow::kSavedModelTagServe}, 
    &model);
  if (!load_status.ok()) {
    LOG(ERROR) << "LoadSavedModel failed: " << load_status;
    return -1;
  }

  std::unordered_set<std::string> input_names, output_names;
  tensorflow::GetSignatureDefsInputsAndOutputs(model, &input_names, &output_names);
  for (auto it: input_names) {
    LOG(INFO) << "input_names: " << it;
  }
  for (auto it: output_names) {
    LOG(INFO) << "output_names: " << it;
  }

  if (verbose) {
    auto meta_graph_def = model.meta_graph_def;
    LOG(INFO) << "meta_graph_version: " << meta_graph_def.meta_info_def().meta_graph_version();
    LOG(INFO) << "tensorflow_version: " << meta_graph_def.meta_info_def().tensorflow_version();
    LOG(INFO) << "tensorflow_git_version: " << meta_graph_def.meta_info_def().tensorflow_git_version();
    for (auto tag: meta_graph_def.meta_info_def().tags()) {
      LOG(INFO) << "tags: " << tag;
    }
    for (auto alias: meta_graph_def.meta_info_def().function_aliases()) {
      LOG(INFO) << "function_alias: " << alias.first << ": " << alias.second;
    }
    for (auto sign: meta_graph_def.signature_def()) {
      //LOG(INFO) << "signature_def: " << it.first << ": " << it.second;
      LOG(INFO) << "signature_def: " << sign.first;
      LOG(INFO) << "\tmethod_name: " << sign.second.method_name();
      LOG(INFO) << "\tinputs: ";
      for (auto it: sign.second.inputs()) {
        LOG(INFO) << "\t\tkey: " << it.first
                  << ", name: " << it.second.name()
                  << ", dtype: " << tensorflow::DataTypeString(it.second.dtype())
                  << ", tensor_shape: " << tensorflow::cc_op::PrintTensorShape(it.second.tensor_shape());
      }
      LOG(INFO) << "\toutputs: ";
      for (auto it: sign.second.outputs()) {
        LOG(INFO) << "\t\tkey: " << it.first
                  << ", name: " << it.second.name()
                  << ", dtype: " << tensorflow::DataTypeString(it.second.dtype())
                  << ", tensor_shape: " << tensorflow::cc_op::PrintTensorShape(it.second.tensor_shape());
      }
    }
  }

  // Actually run the image through the model.
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = model.GetSession()->Run({{input_layer, resized_tensor}}, {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  if (verbose) {
    // Do something interesting with the results we've generated.
    for (const auto& record : outputs) {
      LOG(INFO) << record.DebugString();
      LOG(INFO) << "DataType: " << tensorflow::DataTypeString(record.dtype()) << ", NumElements: " << record.NumElements();
      int dims = record.dims();
      LOG(INFO) << "dimensions: " << dims;
      for (int i=0; i<dims; ++i) {
        LOG(INFO) << "dimensions[" << i << "]=" << record.dim_size(i);
      }
      LOG(INFO) << "tensor values: ";
      // outputs[0] is a 1x10 matrix
      auto tm = outputs[0].matrix<float>();
      for (int i=0; i<outputs[0].dim_size(0); ++i) {
        for (int j=0; j<outputs[0].dim_size(1); ++j) {
          LOG(INFO) << "value[" << i << "," << j << "]=" << tm(i, j);
        }
      }
    }
  }

  int predicted_label = -1;
  run_status = GetPredictedLabel(outputs, predicted_label, verbose);
  if (!run_status.ok()) {
    LOG(ERROR) << "GetPredictedLabel failed: " << run_status;
    return -1;
  }
  LOG(INFO) << "predicted result: " << predicted_label;
  return 0;
}
