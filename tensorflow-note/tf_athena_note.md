# Tensorflow at Athena

- graphs

> Taken from *Hands-On Machine Learning with Scikit-Learn & TensorFlow*

> A TensorFlow program is typically split into two parts: the first part builds a computation graph (this is called the **construction** phase), and the second part runs it (this is the **execution** phase). The construction phase typically builds a computation graph representing the ML model and the computations required to train it. The execution phase generally runs a loop that evaluates a training step repeatedly (for example, one step per mini-batch), gradually improving the model parameters. We will go through an example shortly.

- gradients

```py
help(tf.gradients)
gradients_v2(ys, xs, grad_ys=None, name='gradients', gate_gradients=False, aggregation_method=None, stop_gradients=None, unconnected_gradients=<UnconnectedGradients.NONE: 'none'>)
    Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.
    
    `tf.gradients` is only valid in a graph context. In particular,
    it is valid in the context of a `tf.function` wrapper, where code
    is executing as a graph.
    
    `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
    is a list of `Tensor`, holding the gradients received by the
    `ys`. The list must be the same length as `ys`.
    
    `gradients()` adds ops to the graph to output the derivatives of `ys` with
    respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
    each tensor is the `sum(dy/dx)` for y in `ys` and for x in `xs`.
```

- placeholder

> Taken from *Hands-On Machine Learning with Scikit-Learn & TensorFlow*

> These nodes are special because they don’t actually perform any computation, they just output the data you tell them to output at runtime. They are typically used to pass the training data to TensorFlow during training. If you don’t specify a value at runtime for a placeholder, you get an exception.


- PermissionDeniedError when trying to create a directory

```bash
diff --git a/docker/docker-compose.yml b/docker/docker-compose.yml
index 128db44..50af148 100644
--- a/docker/docker-compose.yml
+++ b/docker/docker-compose.yml
@@ -7,6 +7,7 @@ services:
       args:
         - username=devel
         - userid=1000
+    user: root  # run container as root
     volumes:
       - ../:/home/devel/handson-ml
-    command: /opt/conda/envs/tf1/bin/jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser
+    command: /opt/conda/envs/tf1/bin/jupyter notebook --ip='0.0.0.0' --port=8888 --no-browser --allow-root
```