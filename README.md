# nn.GlobalAttention
Pytorch module for implementing global attention. The original paper implemented this in such a way that the batch size used during training was fixed and permanent, even during testing and deployment. This implementation does not stack the inputs before the attention module and instead lets the module the learn the mask in a batch setting just like every other convolution in the network.

Original paper:
A Novel Global Spatial Attention Mechanism in Convolutional Neural Network for Medical Image Classification.
https://arxiv.org/abs/2007.15897
