# Add Weight Initializations when 
Weights that are initialized to large values can lead to vanishing or exploding gradients, depending on the activation function being used. This can cause the model to converge slowly or not at all. Weights that are initialized to small random values can lead to more efficient training, as the optimization algorithm is able to make larger updates to the weights at the beginning of training. Different initialization methods can be more suitable for different types of problems and model architectures.

## Need for Weight Initializations
- Sigmoid/Tanh: vanishing gradients
    - Constant Variance initialization with Lecun or Xavier
- ReLU: exploding gradients with dead units
    - He Initialization
- Leaky ReLU: exploding gradients only
    - He Initialization
## Types of weight initialisations
- Zero
- Normal: growing weight variance
    - nn.init.normal_(self.fc1.weight, mean=0, std=1)
- Lecun: constant variance
    - by default (no additonal code)
- Xavier: constant variance for Sigmoid/Tanh
    - nn.init.xavier_normal_(self.fc2.weight)
- Kaiming He: constant variance for ReLU activations
    - nn.init.kaiming_normal_(self.fc2.weight)