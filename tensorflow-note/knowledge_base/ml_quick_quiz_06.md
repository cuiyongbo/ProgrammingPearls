```
Q: in pytorch how to save a model after training?

In PyTorch, saving a model after training is a common task that can be accomplished using the `torch.save` function. There are two main ways to save a model in PyTorch:

1. **Saving the entire model**: This saves the model architecture, weights, and other information.
2. **Saving only the model's state_dict**: This saves only the model's parameters (weights and biases), which is the recommended approach.

### Saving the Entire Model

Saving the entire model is straightforward but less flexible. It saves the model architecture along with the weights, which can be useful if you want to reload the model without redefining its architecture.

```python
import torch

# Assuming `model` is your trained model
torch.save(model, 'model.pth')
```

### Loading the Entire Model

```python
import torch

# Load the model
model = torch.load('model.pth')
model.eval()  # Set the model to evaluation mode
```

### Saving Only the Model's state_dict

Saving only the `state_dict` is the recommended approach because it is more flexible and allows you to load the model into different architectures.

#### Saving the state_dict

```python
import torch

# Assuming `model` is your trained model
torch.save(model.state_dict(), 'model_state.pth')
```

#### Loading the state_dict

To load the `state_dict`, you need to first define the model architecture and then load the saved parameters.

```python
import torch
import torch.nn as nn

# Define the model architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = nn.Linear(10, 1)

    def forward(self, x):
        return self.dense(x)

# Create the model instance
model = SimpleModel()

# Load the state_dict
model.load_state_dict(torch.save('model_state.pth'))
model.eval()  # Set the model to evaluation mode
```

### Example: Complete Workflow

Here is a complete example that includes training a simple model, saving the model's state_dict, and then loading it for inference.

#### Training and Saving the Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = nn.Linear(10, 1)

    def forward(self, x):
        return self.dense(x)

# Create the model instance
model = SimpleModel()

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Save the model's state_dict
torch.save(model.state_dict(), 'model_state.pth')
```

#### Loading the Model for Inference

```python
import torch
import torch.nn as nn

# Define the model architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = nn.Linear(10, 1)

    def forward(self, x):
        return self.dense(x)

# Create the model instance
model = SimpleModel()

# Load the state_dict
model.load_state_dict(torch.load('model_state.pth'))
model.eval()  # Set the model to evaluation mode

# Dummy input for inference
input_data = torch.randn(1, 10)
output = model(input_data)
print("Output:", output)
```

### Summary

- **Saving the Entire Model**: Use `torch.save(model, 'model.pth')` to save the entire model.
- **Saving the state_dict**: Use `torch.save(model.state_dict(), 'model_state.pth')` to save only the model's parameters.
- **Loading the Entire Model**: Use `model = torch.load('model.pth')` to load the entire model.
- **Loading the state_dict**: Define the model architecture, create an instance, and use `model.load_state_dict(torch.load('model_state.pth'))` to load the parameters.

By following these steps, you can effectively save and load your PyTorch models for future use.

Q: what are the differences between resnet50, resnet101, resnet1152?

ResNet (Residual Network) is a popular deep learning architecture introduced by Kaiming He et al. in their 2015 paper "Deep Residual Learning for Image Recognition." The key idea behind ResNet is the use of residual connections, which help mitigate the vanishing gradient problem and allow for the training of very deep networks.

### Differences Between ResNet Variants

The primary difference between ResNet50, ResNet101, and ResNet152 lies in the depth of the network, i.e., the number of layers. Here's a detailed comparison:

1. **ResNet50**:
   - **Number of Layers**: 50
   - **Architecture**: Consists of 49 convolutional layers and 1 fully connected layer.
   - **Building Blocks**: Uses a combination of 1x1, 3x3, and 1x1 convolutional layers in its residual blocks.
   - **Performance**: Provides a good balance between computational efficiency and accuracy, making it a popular choice for many applications.

2. **ResNet101**:
   - **Number of Layers**: 101
   - **Architecture**: Consists of 100 convolutional layers and 1 fully connected layer.
   - **Building Blocks**: Similar to ResNet50 but with more layers, allowing for a deeper network that can capture more complex features.
   - **Performance**: Generally offers higher accuracy than ResNet50 but at the cost of increased computational complexity and memory usage.

3. **ResNet152**:
   - **Number of Layers**: 152
   - **Architecture**: Consists of 151 convolutional layers and 1 fully connected layer.
   - **Building Blocks**: Similar to ResNet101 but with even more layers, making it one of the deepest ResNet variants.
   - **Performance**: Typically provides the highest accuracy among the three variants but requires significantly more computational resources and memory.

### Detailed Architecture

#### ResNet50

- **Convolutional Layers**: 49
- **Fully Connected Layer**: 1
- **Residual Blocks**: Uses bottleneck blocks with 1x1, 3x3, and 1x1 convolutions.
- **Structure**:
  - Initial convolution and max-pooling layers.
  - 3 stages of residual blocks with 3 layers each.
  - 4 stages of residual blocks with 4 layers each.
  - 6 stages of residual blocks with 6 layers each.
  - 3 stages of residual blocks with 3 layers each.
  - Global average pooling and fully connected layer.

#### ResNet101

- **Convolutional Layers**: 100
- **Fully Connected Layer**: 1
- **Residual Blocks**: Uses bottleneck blocks with 1x1, 3x3, and 1x1 convolutions.
- **Structure**:
  - Initial convolution and max-pooling layers.
  - 3 stages of residual blocks with 3 layers each.
  - 4 stages of residual blocks with 4 layers each.
  - 23 stages of residual blocks with 23 layers each.
  - 3 stages of residual blocks with 3 layers each.
  - Global average pooling and fully connected layer.

#### ResNet152

- **Convolutional Layers**: 151
- **Fully Connected Layer**: 1
- **Residual Blocks**: Uses bottleneck blocks with 1x1, 3x3, and 1x1 convolutions.
- **Structure**:
  - Initial convolution and max-pooling layers.
  - 3 stages of residual blocks with 3 layers each.
  - 8 stages of residual blocks with 8 layers each.
  - 36 stages of residual blocks with 36 layers each.
  - 3 stages of residual blocks with 3 layers each.
  - Global average pooling and fully connected layer.

### Performance and Trade-offs

- **Accuracy**: Generally, deeper networks like ResNet101 and ResNet152 offer higher accuracy compared to ResNet50. However, the improvement in accuracy diminishes as the network depth increases.
- **Computational Complexity**: Deeper networks require more computational resources and memory, making them more challenging to deploy on resource-constrained devices.
- **Training Time**: Deeper networks take longer to train due to the increased number of layers and parameters.

### Summary

- **ResNet50**: 50 layers, good balance between accuracy and computational efficiency.
- **ResNet101**: 101 layers, higher accuracy but increased computational complexity.
- **ResNet152**: 152 layers, highest accuracy among the three but requires significantly more resources.

By understanding the differences between ResNet50, ResNet101, and ResNet152, you can choose the appropriate variant based on your specific requirements, such as accuracy, computational resources, and deployment constraints.

Q: how to display an image in jupyter notebook?

* with PIL you can

```python
img =  Image.open('path-to-image-file')
display(img)
```

* convert PIL.image to pytorch.tensor

```python
img =  Image.open('path-to-image-file')
# img_tensor = torchvision.transforms.functional.pil_to_tensor(img)
# or 
to_tensor = torchvision.transforms.PILToTensor()
img_tensor = to_tensor(img)
```

* convert pytorch.tensor to PIL.image 

```python
to_pil = torchvision.transforms.ToPILImage()
img = to_pil(your-tensor)
```
```