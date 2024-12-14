#include <stdio.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"


void hello_tf_v1() {
  LOG(INFO) << "running " << __func__ << " ##################";

  // create a root scope
  auto scope = tensorflow::Scope::NewRootScope();

  // define various constans/inputs on which we
  // will perform an operation
  auto hello = tensorflow::ops::Const(scope, std::string("hello"));
  auto space = tensorflow::ops::Const(scope, std::string(" "));
  auto world = tensorflow::ops::Const(scope, std::string("world"));

  // StringJoin operation
  auto joinOp = tensorflow::ops::StringJoin(scope, {hello, space, world});

  // create a session that takes our
  // scope as the root scope
  tensorflow::ClientSession session(scope);

  // Run
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session.Run({joinOp}, &outputs));

  // See our output using DebugString that tells
  // more information about the tensor
  std::cout << "DebugString -> " << outputs[0].DebugString() << std::endl;
  // we can also get the underlying data by calling flat
  std::cout << "Underlying Scalar value -> " << outputs[0].flat<tensorflow::tstring>()(0) << std::endl;
}


tensorflow::Status hello_tf_v2() {
  LOG(INFO) << "running " << __func__ << " ##################";


  auto root = tensorflow::Scope::NewRootScope();
  auto hello = tensorflow::ops::Const(root.WithOpName("hello"), std::string("hello"));
  auto space = tensorflow::ops::Const(root.WithOpName("space"), std::string(" "));
  auto world = tensorflow::ops::Const(root.WithOpName("world"), std::string("world"));
  auto joinOp = tensorflow::ops::StringJoin(root.WithOpName("joinOp"), {hello, space, world});  
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run({}, {"joinOp"}, {}, &outputs));

  auto run_status = ::tensorflow::OkStatus();
  if (run_status.ok()) {
    std::cout << "DebugString -> " << outputs[0].DebugString() << std::endl;
    std::cout << "Underlying Scalar value -> " << outputs[0].flat<tensorflow::tstring>()(0) << std::endl;
  }
  return run_status;
}


void hello_tf_v3() {
  LOG(INFO) << "running " << __func__ << " ##################";
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  // 2x2 Matrix A = [3 2; -1 0]
  auto A = tensorflow::ops::Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // 1x2 Matrix b = [3 5]
  auto b = tensorflow::ops::Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = tensorflow::ops::MatMul(root.WithOpName("v"), A, b, tensorflow::ops::MatMul::TransposeB(true));
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();

  auto u = tensorflow::ops::MatMul(root.WithOpName("u"), b, A);
  TF_CHECK_OK(session.Run({u}, &outputs));
  LOG(INFO) << outputs[0].matrix<float>();
}


int hello_tf_v4() {
  LOG(INFO) << "running " << __func__ << " ##################";
  auto scope = tensorflow::Scope::NewRootScope();
  auto a = tensorflow::ops::Const(scope.WithOpName("A"), 23);
  auto b = tensorflow::ops::Const(scope.WithOpName("B"), 2);
  auto add = tensorflow::ops::Add(scope.WithOpName("Add"), a, b);
  auto sub = tensorflow::ops::Sub(scope.WithOpName("Sub"), add, b);
  tensorflow::ClientSession session(scope);
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session.Run({add, sub}, &outputs));
  std::cout << "Outputs size: " << outputs.size() << std::endl;
  std::cout << "DebugString Add: " << outputs[0].DebugString() << std::endl;
  std::cout << "DebugString Sub: " << outputs[1].DebugString() << std::endl;

  outputs.clear();

  TF_CHECK_OK(session.Run({sub, add}, &outputs));
  std::cout << "Outputs size: " << outputs.size() << std::endl;
  std::cout << "DebugString Add: " << outputs[1].DebugString() << std::endl;
  std::cout << "DebugString Sub: " << outputs[0].DebugString() << std::endl;

  return 0;
}


int hello_tf_v5() {
  LOG(INFO) << "running " << __func__ << " ##################";
  {  
    auto scope = tensorflow::Scope::NewRootScope();
    auto a = tensorflow::ops::Const(scope.WithOpName("A"), 23);
    auto b = tensorflow::ops::Const(scope.WithOpName("B"), 2);
    auto add = tensorflow::ops::Add(scope.WithOpName("Add"), a, b);
    auto sub = tensorflow::ops::Sub(scope.WithOpName("Sub"), add, b);
    tensorflow::GraphDef graph;
    scope.ToGraphDef(&graph);
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session->Create(graph));
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session->Run({}, {"Sub"}, {}, &outputs));
    std::cout << "Outputs size: " << outputs.size() << std::endl;
    std::cout << "DebugString Sub: " << outputs[0].DebugString() << std::endl;

    TF_CHECK_OK(session->Run({}, {"Sub", "Add"}, {}, &outputs));
    std::cout << "Outputs size: " << outputs.size() << std::endl;
    std::cout << "DebugString Add: " << outputs[1].DebugString() << std::endl;
    std::cout << "DebugString Sub: " << outputs[0].DebugString() << std::endl;
  }

  {  
    auto scope = tensorflow::Scope::NewRootScope();
    auto a = tensorflow::ops::PlaceholderWithDefault(scope.WithOpName("A"), tensorflow::Input(8), tensorflow::PartialTensorShape());
    auto b = tensorflow::ops::Placeholder(scope.WithOpName("B"), tensorflow::DT_INT32);
    auto add = tensorflow::ops::Add(scope.WithOpName("Add"), a, b);
    tensorflow::GraphDef graph;
    scope.ToGraphDef(&graph);
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session->Create(graph));
    // prepare input
    auto input_b = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
    input_b.scalar<int>()() = 20;
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session->Run({{"B", input_b}}, {"Add"}, {}, &outputs));
    std::cout << "Outputs size: " << outputs.size() << std::endl;
    std::cout << "DebugString Add: " << outputs[0].DebugString() << std::endl;

    auto input_a = tensorflow::Tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
    input_a.scalar<int>()() = 60;
    TF_CHECK_OK(session->Run({{"B", input_b}, {"A", input_a}}, {"Add"}, {}, &outputs));
    std::cout << "Outputs size: " << outputs.size() << std::endl;
    std::cout << "DebugString Add: " << outputs[0].DebugString() << std::endl;
  }

  return 0;
}


int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());

  hello_tf_v1();
  hello_tf_v2();
  hello_tf_v3();
  hello_tf_v4();
  hello_tf_v5();

  return 0;
}
