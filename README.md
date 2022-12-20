# Activation Learning

This project implements activation learning ([https://arxiv.org/abs/2209.13400](https://arxiv.org/abs/2209.13400)) on the MNIST and CIFAR-10 datasets.
Activation learning supports bottom-up unsupervised learning based on a simple local learning rule. In activation learning, the output activation (sum of the squared output) of the neural network estimates the likelihood of the input patterns, or ``learn more,
activate more" in simpler terms.


# Classification on MNIST

Activation learning achieves about a 1.36% test error rate on MNIST with a two-layer network, comparable to a baseline of backpropagation when
using a fully connected network without complicated regularizers. As the number of training samples reduces, activation learning can outperform backpropagation.
Using 600 labeled training samples from MNIST, activation learning can reach a classification error rate of 9.74%,
while backpropagation can only get an error rate of 16.17% with one additional output layer.

#### Classification without feedback (only positive training samples):
```commandline
python3 -u fully_connected_activation.py  --feature_layers=2 --log_dir=model
```

#### Classification with feedback (positive and negative training samples):
```commandline
python3 -u fully_connected_activation.py  --feature_layers=2 --log_dir=model --feedback
```

#### BackPropagation:
```commandline
python3 -u fully_connected_bp.py  --feature_layers=2 --log_dir=model
```

# Robustness of Learning

Activation learning is shown more robust against external disturbances than backpropagation on MNIST. In the experiments, we mark
the bottom portions of the test images or add noisy lines to the test images. In a wide range, the error rate of activation learning 
is roughly half that of backpropagation (with a fully connected network).

#### Classification on masked images:
```commandline
python3 -u inference_masked.py  --feature_layers=2 --log_dir=model --experiments5
```

#### Classification on lined images:
```commandline
python3 -u inference_lined.py  --feature_layers=2 --log_dir=model --experiments6
```

# Image Generation

In activation learning, the same trained neural network for classification can be used for image generation and image completions.
A discriminative task is similar to a generative task in that the objective of both is to find missing units that maximize output activation.

#### Image generation from given classes
```commandline
python3 -u generative_model.py  --feature_layers=2 --log_dir=model --add_noise
```

#### Image completion from top half images
```commandline
python3 -u completion_model.py  --feature_layers=2 --log_dir=model 
```

# Classification on CIFAR-10

A network with local connections (expect the top layer) is used for experiments on CIFAR-10. 
Without data augmentation, activation learning can achieve about a 37.14% test error rate, comparable to 
the 36.63% error rate of backpropagation based on about the same network (with 10 output units on top). However, when data augmentation is applied, 
activation learning currently cannot compete with backpropagation on CIFAR-10, and it needs further study.

#### Classification without feedback
```commandline
python3 -u local_net_validation.py  --feature_layers=3 --kernel_size=5 --log_dir=model --nofeedback
```

#### Classification with feedback:
```commandline
python3 -u local_net_activation.py  --feature_layers=2 --kernel_size=5 --log_dir=model
```

#### Classification by backpropagation:
```commandline
python3 -u local_net_bp.py  --feature_layers=2 --log_dir=model
```

#### Parameters:

```
--feature_layers, --training_size, --kernel_size(receptive field), --channels, --unlearning_factor, --nofeedback
```

Some places to change in the files:
```python
# Applying data augmentation(local_net_activation.py, local_net_bp.py, local_net_validation.py)
import cifar10_input    
# import cifar10_input_noaug as cifar10_input

# weight decay if no feedback (cifar10_inference.py)
delta = delta - weight * 1e-5
```

