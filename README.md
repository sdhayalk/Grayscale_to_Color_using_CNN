# Grayscale to Color using CNN
## Automatic Colorization of Grayscale images using Deep Learning

Developed a Convolutional Neural Network architecture to generate Color images from Grayscale input images.

Specifically, it generates color images in the YUV channels space when given input grayscale images in the Y channel.

Implemented a modified version of VGG-16 along with use of its Hypercolumns to concatenate with upscaling Hypercolumns (inspired from residual skip connections in ResNet) to generate UV channel output.

Trained and tested on cloud with Places dataset using Huber Loss function and Adam Optimizer.

One of the issue is that it sometimes produces sepia toned images. This is because of the simple loss function
## Dataset
Places dataset. [More details here](http://places.csail.mit.edu/)

## TODO
1. Implement a better and complex loss function
2. Provide sample images for input, ground truth and predicted
