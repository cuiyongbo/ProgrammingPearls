#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/platform/env.h>


void basic_sesson_ops() {
    auto scope = tensorflow::Scope::NewRootScope();

    {    
        tensorflow::ClientSession session(scope);
        auto a = tensorflow::ops::Const(scope, 2);
        auto b = tensorflow::ops::Const(scope, 2);
        auto c = tensorflow::ops::Add(scope, a, b);
        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session.Run({c}, &outputs)); // you may not change input
        std::cout << "Add(a, b) = " << outputs[0].flat<int>() << std::endl;
    }

    {
        tensorflow::ClientSession session(scope);
        auto a = tensorflow::ops::Placeholder(scope, tensorflow::DT_INT32);
        auto b = tensorflow::ops::Placeholder(scope, tensorflow::DT_INT32);
        auto c = tensorflow::ops::Add(scope, a, b);
        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session.Run({{{a, 2}, {b, 3}}}, {c}, &outputs)); // specify input at run-time
        std::cout << "Add(a, b) = " << outputs[0].flat<int>() << std::endl;
    }

    {
        tensorflow::ClientSession session(scope);
        auto defaultAInput = tensorflow::Input(8);
        auto a = tensorflow::ops::PlaceholderWithDefault(scope, defaultAInput, tensorflow::PartialTensorShape());
        auto b = tensorflow::ops::Placeholder(scope, tensorflow::DT_INT32);
        auto c = tensorflow::ops::Add(scope, a, b);
        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session.Run({{{b, 3}}}, {c}, &outputs)); // use default value if input a is unspecified
        std::cout << "Add(a, b) = " << outputs[0].flat<int>() << std::endl;
        TF_CHECK_OK(session.Run({{{a, 4}, {b, 3}}}, {c}, &outputs));
        std::cout << "Add(a, b) = " << outputs[0].flat<int>() << std::endl;
    }
}


int basic_io_ops() {
    std::string file_path = "./data/scaffold.txt";
    auto env = tensorflow::Env::Default();
    auto status = env->FileExists(file_path);
    if (!status.ok()) {
        std::cerr << file_path << " does not exist" << std::endl;
        return - 1;
    }

    tensorflow::uint64 file_size;
    status = env->GetFileSize(file_path, &file_size);
    if (!status.ok()) {
        std::cerr << "failed to get the size of " << file_path << std::endl;
        return -1;
    }

    std::cout << "file size of " << file_path << ": " << file_size << std::endl;
    std::unique_ptr<tensorflow::RandomAccessFile> file_obj;
    status = env->NewRandomAccessFile(file_path, &file_obj);
    if (!status.ok()) {
        std::cerr << "failed to open " << file_path << std::endl;
        return -1;
    }

    std::string content;
    content.resize(file_size);
    tensorflow::StringPiece sp;
    file_obj->Read(0, file_size, &sp, &(content)[0]);
    if (!status.ok()) {
        std::cerr << "failed to read " << file_path << std::endl;
        return -1;
    }
    std::cout << content << std::endl;
    tensorflow::Tensor result(tensorflow::DT_STRING, tensorflow::TensorShape());
    //result.scalar<std::string>()() = sp.ToString();
    result.scalar<tensorflow::tstring>()() = content;

    return 0;
}


int main(int argc, char **argv) {
    basic_sesson_ops();
    basic_io_ops();
    return 0;
}
