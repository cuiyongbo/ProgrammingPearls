# Tensorflow Quiz - interview-oriented

```

Q: what are frequently asked questions when I prepare an interview for model inference, especially for tensorflow model?

When preparing for an interview focused on model inference, especially with TensorFlow models, you might encounter a variety of questions that test your understanding of the TensorFlow framework, model deployment, optimization, and general machine learning concepts. Here are some frequently asked questions you might come across:

### TensorFlow-Specific Questions
1. What is TensorFlow, and how does it differ from other deep learning frameworks?
2. Can you explain the difference between TensorFlow 1.x and TensorFlow 2.x?
3. What are TensorFlow graphs, and how do they work?
4. How do you create a TensorFlow dataset for training or inference?
5. What is the purpose of TensorFlow Serving, and how does it help with model inference?
6. How do you convert a trained TensorFlow model to TensorFlow Lite for mobile or edge devices?
7. What are TensorFlow Estimators, and when would you use them?
8. How do you use TensorFlow's `tf.function` decorator, and what are its benefits?

### Model Inference and Deployment Questions
9. What is model inference, and how does it differ from model training?
10. How do you optimize a TensorFlow model for faster inference?
11. What are some common challenges you might face when deploying a model for inference, and how would you address them?
12. Can you explain the concept of batch inference, and when it might be used?
13. What are the considerations for deploying a TensorFlow model in a production environment?
14. How do you monitor and ensure the performance of a model in production?
15. What is A/B testing in the context of model deployment, and why is it important?

### Performance Optimization Questions
16. What techniques can you use to reduce the size of a TensorFlow model?
17. How do you use quantization and pruning to optimize a TensorFlow model?
18. What is the role of hardware accelerators like GPUs and TPUs in model inference?
19. Can you explain the concept of parallelism and distribution in TensorFlow?
20. How do you handle real-time inference requirements in TensorFlow?

### General Machine Learning Questions
21. What are precision and recall, and how do they relate to model inference?
22. How do you evaluate the performance of a model during inference?
23. What is overfitting, and how can it affect inference results?
24. Can you explain the difference between supervised and unsupervised learning, and how does it impact model inference?
25. What are some common data preprocessing steps required before performing inference?

### Behavioral Questions
26. Describe a challenging model deployment you worked on. What was the problem, and how did you solve it?
27. How do you stay updated with the latest advancements in TensorFlow and machine learning?
28. Can you discuss a time when you had to optimize a model for better inference performance? What steps did you take?

Remember that the specific questions you'll face can vary depending on the role and the company. It's also important to be prepared to discuss any relevant projects or experiences you have that demonstrate your skills in TensorFlow and model inference.


1. What is TensorFlow, and how does it differ from other deep learning frameworks?

TensorFlow is an open-source machine learning library developed by the Google Brain team. It is widely used for building and deploying machine learning models, particularly deep learning models. TensorFlow provides a comprehensive ecosystem of tools, libraries, and community resources that enable researchers and developers to build and deploy ML-powered applications.

Key features of TensorFlow include:

- **Graph-based execution:** TensorFlow models consist of a dataflow graph where nodes represent operations, and edges represent the data consumed or produced by these operations.
- **Eager execution:** TensorFlow 2.x supports eager execution by default, which allows operations to be evaluated immediately without building graphs. This makes the framework more intuitive and easier to debug.
- **Scalability:** TensorFlow can scale across multiple CPUs and GPUs, making it suitable for a range of devices from mobile phones to large-scale distributed systems.
- **Flexibility:** TensorFlow supports a wide array of algorithms and can be used for various tasks beyond deep learning, such as reinforcement learning and natural language processing.
- **TensorFlow Serving:** A system for serving machine learning models that is designed for production environments and supports model versioning for easy rollback and canarying new models.
- **TensorFlow Lite:** A lightweight solution for deploying models on mobile and IoT devices.
- **TensorFlow.js:** A library for training and deploying models in the browser and on Node.js.
- **TensorBoard:** A visualization tool for understanding, debugging, and optimizing TensorFlow programs.

Differences from other deep learning frameworks:

- **PyTorch:** Developed by Facebook's AI Research lab, PyTorch is another popular deep learning framework that emphasizes dynamic computation graphs and "define by run" philosophy, which can be more intuitive for developers coming from a Python background. PyTorch is known for its ease of use and simplicity in prototyping, which contrasts with TensorFlow's more verbose and complex graph abstraction (especially in TensorFlow 1.x).
- **Keras:** Initially a separate high-level neural networks library, Keras is now integrated into TensorFlow as `tf.keras`. It provides a more user-friendly API for building models compared to TensorFlow's lower-level API. Keras can also run on top of other frameworks like Theano or Microsoft Cognitive Toolkit (CNTK).
- **Caffe/Caffe2:** Developed by the Berkeley Vision and Learning Center, Caffe is known for its speed and its focus on convolutional neural networks. Caffe2, which merged with PyTorch, is optimized for large-scale industrial applications and mobile deployment.
- **MXNet:** Apache MXNet is designed for both efficiency and flexibility. It allows developers to mix symbolic and imperative programming to maximize efficiency and productivity. MXNet is also part of the Amazon Web Services (AWS) ecosystem.

Each framework has its strengths and is chosen by developers and researchers based on specific needs, such as ease of use, performance, scalability, and the specific tools and features they offer. TensorFlow's broad adoption, extensive documentation, and strong community support make it a go-to choice for many machine learning tasks.

2. Can you explain the difference between TensorFlow 1.x and TensorFlow 2.x?

TensorFlow 1.x and TensorFlow 2.x are two major versions of the TensorFlow machine learning library, with significant differences in their design, API, and usability. Here are the key differences between the two:

### TensorFlow 1.x:

- **Graph-based execution:** TensorFlow 1.x relies heavily on a static computation graph, which means that the computational operations and the data flow between them are defined in advance and then executed. This approach can be less intuitive and harder to debug because the graph must be fully defined before running the session to execute it.
- **Sessions:** To execute the graph, TensorFlow 1.x requires the use of a `Session` object. This means that you need to create a session and explicitly run the computational graph within this session to get results.
- **Verbose and complex:** TensorFlow 1.x's API can be verbose and complex, especially for beginners. It requires users to manage a lot of boilerplate code and to explicitly handle placeholders and variables.
- **Scope-based variable sharing:** Sharing variables in TensorFlow 1.x often involves using `tf.variable_scope` and `tf.get_variable`, which can be cumbersome and error-prone.

### TensorFlow 2.x:

- **Eager execution by default:** TensorFlow 2.x enables eager execution by default, which means that operations are evaluated immediately without the need to build and run a graph. This makes the development process more intuitive and similar to regular Python programming.
- **Simplified API:** TensorFlow 2.x has a more streamlined and user-friendly API, with many convenience functions that reduce boilerplate code. It integrates Keras as the high-level API for building and training models, making it easier to get started with deep learning.
- **Functions, not sessions:** With the introduction of `tf.function`, TensorFlow 2.x allows you to convert Python functions into graph-executable TensorFlow operations, which can be optimized for performance. This eliminates the need for sessions and makes the code cleaner and more Pythonic.
- **Improved variable management:** TensorFlow 2.x simplifies variable management by using Python's native support for variables, which means you no longer need to use variable scopes for sharing variables.
- **Better integration and consistency:** TensorFlow 2.x provides better integration with the TensorFlow ecosystem, including TensorFlow Extended (TFX), TensorFlow Lite, TensorFlow.js, and TensorFlow Hub, offering a more consistent experience across different platforms and applications.

Overall, TensorFlow 2.x represents a significant shift towards a more user-friendly and Pythonic approach to deep learning, with a focus on simplicity, ease of use, and flexibility. It retains the scalability and performance of TensorFlow 1.x while making it more accessible to a broader audience of developers and researchers.


Q: what is the difference between ``torch.sum(a, 0)`` and ``torch.sum(a, 1)``?

```py
>>> a = torch.tril(torch.ones(3,3))
>>> a
tensor([[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])

>>> a0 = torch.sum(a, 0, keepdim=True)
>>> a0.shape
torch.Size([1, 3])
>>> a0
tensor([[3., 2., 1.]])
>>> a1 = torch.sum(a, 1, keepdim=True)
>>> a1.shape
torch.Size([3, 1])
>>> a1
tensor([[1.],
        [2.],
        [3.]])
>>> help(torch.sum)
    If :attr:`keepdim` is ``True``, the output tensor is of the same size
    as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
    Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
    output tensor having 1 (or ``len(dim)``) fewer dimension(s).
```

```