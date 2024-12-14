#include  <algorithm>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"


void tensor_ops() {
    LOG(INFO) << "running " << __func__ << " ##################";
    auto scope = tensorflow::Scope::NewRootScope();

    { // naive operation
        auto aScalar = tensorflow::Input(2);
        std::cout << "Dimensions of a scalar: " << aScalar.tensor().shape().dims() << std::endl;
        auto aVector = tensorflow::Input({2,3});
        std::cout << "Dimensions of a vector: " << aVector.tensor().shape().dims() << std::endl;
        auto aMatrix = tensorflow::Input({{2,3}, {7, 8}});
        std::cout << "Dimensions of a matrix: " << aMatrix.tensor().shape().dims() << std::endl;
    }

    { // scalar
        tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({}));
        auto Tt = t.scalar<float>();
        t.scalar<float>()() = 123.45f;
        LOG(INFO) << "NumElements: " << t.NumElements()
                  << ", rank: " << t.dims()
                  << ", shape: " << t.shape()
                  << ", dataType: " << tensorflow::DataTypeString(t.dtype())
                  << ", value: " << Tt();
    }

    { // vector
        tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
        auto Tt = t.vec<float>();
        t.vec<float>()(0) = 123.45f;
        LOG(INFO) << "NumElements: " << t.NumElements()
                  << ", rank: " << t.dims()
                  << ", shape: " << t.shape()
                  << ", dataType: " << tensorflow::DataTypeString(t.dtype())
                  << ", value: " << Tt(0);
    }

    { // matrix
        tensorflow::Tensor record(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 4}));
        record.flat<float>().setRandom();
        // Do something interesting with the results we've generated.
        LOG(INFO) << record.DebugString();
        LOG(INFO) << "DataType: " << tensorflow::DataTypeString(record.dtype()) 
                  << "TensorShape: " << record.shape() 
                  << ", NumElements: " << record.NumElements();
        int dims = record.dims();
        LOG(INFO) << "dimensions: " << dims;
        for (int i=0; i<dims; ++i) {
            LOG(INFO) << "dimensions[" << i << "]=" << record.dim_size(i);
        }
        LOG(INFO) << "tensor values: ";
        auto tm = record.matrix<float>();
        for (int i=0; i<record.dim_size(0); ++i) {
            for (int j=0; j<record.dim_size(1); ++j) {
                LOG(INFO) << "value[" << i << "," << j << "]=" << tm(i, j);
            }
        }
    }

    {
        tensorflow::ClientSession session(scope);
        // 2x2 matrix with all elements = 10
        //auto c1 = tensorflow::ops::Const(scope, 10, {2,2}/*shape*/);
        // how to print a Const ??
        //std::cout << "Const value: " << c1.matrix<int>() << std::endl;
        auto x = tensorflow::ops::MatMul(scope, {{1, 1}}, {{41}, {1}});
        std::vector<tensorflow::Tensor> outputs;
        auto status = session.Run({x}, &outputs);
        TF_CHECK_OK(status);
        std::cout << "MatMul result: " << outputs[0].flat<int>() << std::endl;
        auto y = tensorflow::ops::Add(scope, {1,2,3,4}, 10);
        status = session.Run({y}, &outputs);
        TF_CHECK_OK(status);
        std::cout << "Add result: " << outputs[0].flat<int>() << std::endl;
    }
}


tensorflow::Status argmax_op(bool verbose=false) {
    LOG(INFO) << "running " << __func__ << " ##################";
    tensorflow::Tensor dummy(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 4}));
    dummy.flat<float>().setRandom();
    auto root = tensorflow::Scope::NewRootScope();
    std::string output_name = "arg_max";
    // reduce tensor according to axis 1
    tensorflow::ops::ArgMax(root.WithOpName(output_name), dummy, 1);
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    std::vector<tensorflow::Tensor> out_tensors;
    TF_RETURN_IF_ERROR(session->Run({}, {output_name+":0"}, {}, &out_tensors));
    auto dummy_data = dummy.matrix<float>();
    if (verbose) {
        for (int i=0; i<dummy.dim_size(0); ++i) {
            for (int j=0; j<dummy.dim_size(1); ++j) {
                LOG(INFO) << "dummy[" << i << "," << j << "]" << dummy_data(i, j);
            }
        }
        for (const auto& record : out_tensors) {
            LOG(INFO) << "out_tensor TensorShape: " << record.shape()
                      << ", NumElements: " << record.NumElements()
                      << ", DebugString: " << record.DebugString();
        }
    }
    auto record = out_tensors[0];
    auto record_data = record.vec<tensorflow::int64>();
    for (int i=0; i<record.NumElements(); ++i) {
        LOG(INFO) << "Maximum of row[" << i << "]: " << dummy_data(i, record_data(i));
    }
    return tensorflow::OkStatus();
}


tensorflow::Status topk_op(int topk=5, bool verbose=false) {
    LOG(INFO) << "running " << __func__ << " ##################";
    tensorflow::Tensor dummy(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 10}));
    dummy.flat<float>().setRandom();

    topk = std::min(topk, (int)dummy.dim_size(1));

    auto root = tensorflow::Scope::NewRootScope();
    std::string output_name = "top_k";
    tensorflow::ops::TopK(root.WithOpName(output_name), dummy, topk);
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    // The TopK node returns two outputs, the scores and their original indices,
    // so we have to append :0 and :1 to specify them both.
    std::vector<tensorflow::Tensor> out_tensors;
    TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"}, {}, &out_tensors));

    auto dummy_data = dummy.matrix<float>();
    if (verbose) {
        for (int i=0; i<dummy.dim_size(0); ++i) {
            for (int j=0; j<dummy.dim_size(1); ++j) {
                LOG(INFO) << "dummy[" << i << "," << j << "]" << dummy_data(i, j);
            }
        }
        for (const auto& record : out_tensors) {
            LOG(INFO) << "out_tensor TensorShape: " << record.shape()
                      << ", NumElements: " << record.NumElements()
                      << ", DebugString: " << record.DebugString();
        }
    }

    auto scores = out_tensors[0];
    auto scores_data = scores.matrix<float>();
    auto indices = out_tensors[1];
    auto indices_data = indices.matrix<tensorflow::int32>();
    for (int i=0; i<dummy.dim_size(0); ++i) {
        for (int j=0; j<topk; ++j) {
            LOG(INFO) << "top" << topk << " elements of row[" << i << "]: " << dummy_data(i, indices_data(i, j)) << " " << scores_data(i, j);
        }
    }

    return tensorflow::OkStatus();
}


int main(int argc, char **argv) {
    tensor_ops();
    argmax_op(false);
    topk_op(5, false);
    return 0;
}
