# Bazel FAQs

* display help message: ``bazel help sub_command``
* how to uninstall bazel

```bash
rm -rf $HOME/.cache/bazel
sudo rm -rf /usr/local/bin/bazel /etc/bazelrc /usr/local/lib/bazel
```

* list all build targets

```bash
# bazel query //...
//:pip_pkg
//tensorflow_model_optimization:build_docs
//tensorflow_model_optimization:tensorflow_model_optimization
//tensorflow_model_optimization/python:python
//tensorflow_model_optimization/python/core:core
//tensorflow_model_optimization/python/core:version
//tensorflow_model_optimization/python/core/api:api
...
Loading: 35 packages loaded
```

* display more logs when building targets: ``bazel build --verbose_failures //tensorflow:tensorflow_cc``
* how to run test: ``bazel test //tensorflow_model_optimization/python/core/quantization/keras:quantize_functional_test`` (inherit all build opts)

* how can I set CFLAGS in bazel build?

A:To set CFLAGS in a Bazel build, you can use the `--copt` flag when invoking the Bazel build command. The `--copt` flag allows you to pass C/C++ compiler options to the build.
Here's an example of how to set CFLAGS in a Bazel build:

```bash
bazel build --copt=-O2 --copt=-Wall //path/to/package:target
```

In the above command, `--copt=-O2` sets the optimization level to 2, and `--copt=-Wall` enables all warning messages. You can add or modify the `--copt` flags as per your requirements.
If you want to set CFLAGS globally for the entire Bazel workspace, you can create a `.bazelrc` file in the root directory of your project and add the `build --copt` options. For example:

```
build --copt=-O2
build --copt=-Wall
```

This will apply the `-O2` and `-Wall` options to all C/C++ compiler commands throughout the build.
Note that the above examples are for Bazel builds using the command-line interface. If you are using Bazel with a BUILD file, you can set the `copts` attribute in the `cc_binary` or `cc_library` rule to specify the CFLAGS for that specific target.

* Displays information about the state of the bazel process

```sh
> bazel info
DEBUG: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/external/org_tensorflow/third_party/repo.bzl:132:14: 
Warning: skipping import of repository 'icu' because it already exists.
bazel-bin: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/execroot/tf_serving/bazel-out/k8-opt/bin
bazel-genfiles: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/execroot/tf_serving/bazel-out/k8-opt/bin
bazel-testlogs: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/execroot/tf_serving/bazel-out/k8-opt/testlogs
character-encoding: file.encoding = ISO-8859-1, defaultCharset = ISO-8859-1
command_log: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/command.log
committed-heap-size: 153MB
execution_root: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/execroot/tf_serving
gc-count: 28
gc-time: 157ms
install_base: /home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/install/1a80ecdfe78ff11e1566d5084a48c3f7
java-home: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/install/1a80ecdfe78ff11e1566d5084a48c3f7/embedded_tools/jdk
java-runtime: OpenJDK Runtime Environment (build 11.0.6+10-LTS) by Azul Systems, Inc.
java-vm: OpenJDK 64-Bit Server VM (build 11.0.6+10-LTS, mixed mode) by Azul Systems, Inc.
max-heap-size: 4030MB
output_base: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8
output_path: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/execroot/tf_serving/bazel-out
package_path: %workspace%
release: release 5.3.0
repository_cache: /home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/cache/repos/v1
server_log: /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/java.log.n36-182-109.cuiyongbo.log.java.20230721-140424.1201208
server_pid: 1201208
used-heap-size: 69MB
workspace: /data00/home/cuiyongbo/tensorflow/serving
```

*. bazel test to display more information

```bash
#--test_output (summary, errors, all or streamed; default: "summary")
#--test_summary (short, terse, detailed, none or testcase; default: "short")
bazel test --config=release  tensorflow_serving/model_servers:get_model_status_impl_test  --test_output=all
# log file of a test target
# ll bazel-testlogs/tensorflow_serving/model_servers/get_model_status_impl_test/test.log 
-r-xr-xr-x 1 cuiyongbo cuiyongbo 9.1K Jul 23 08:54 bazel-testlogs/tensorflow_serving/model_servers/get_model_status_impl_test/test.log

# cat tensorflow_serving/model_servers/BUILD
cc_test(
    name = "get_model_status_impl_test",
    srcs = ["get_model_status_impl_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_2_versions",
    ],
...

# test_data will be copy to target.runfiles
# cat bazel-bin/tensorflow_serving/model_servers/get_model_status_impl_test.runfiles_manifest
...
tf_serving/tensorflow_serving/model_servers/get_model_status_impl_test /data00/home/cuiyongbo/.cache/bazel/_bazel_cuiyongbo/6690784f550c9cc047d225d7f1dec4b8/execroot/tf_serving/bazel-out/k8-opt/bin/tensorflow_serving/model_servers/get_model_status_impl_test
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/assets/foo.txt /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/assets/foo.txt
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/saved_model.pb /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/saved_model.pb
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/variables/variables.data-00000-of-00001 /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/variables/variables.data-00000-of-00001
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/variables/variables.index /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000123/variables/variables.index
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/assets/foo.txt /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/assets/foo.txt
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/saved_model.pb /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/saved_model.pb
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/variables/variables.data-00000-of-00001 /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/variables/variables.data-00000-of-00001
tf_serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/variables/variables.index /data00/home/cuiyongbo/tensorflow/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions/00000124/variables/variables.index
```


.. rubric:: Footnotes

.. [#] `install bazel <https://bazel.google.cn/install>`_
.. [#] `install bazel on debian 10 <https://tutorialforlinux.com/2018/09/23/bazel-debian-buster-10-installation-guide/2/>`_
.. [#] `bazel installation scripts <https://github.com/bazelbuild/bazel/releases>`_
.. [#] `bazel cpp tutorial <https://bazel.build/start/cpp>`_
.. [#] `bazel build <https://bazel.build/run/build#bazel-build>`_
.. [#] `Write bazelrc configuration files <https://bazel.build/run/bazelrc>`_