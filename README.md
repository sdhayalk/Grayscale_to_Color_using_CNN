# Grayscale to Color using CNN
# Automatic Colorization of Grayscale images using Deep Learning

Developed a Convolutional Neural Network architecture to generate Color images from Grayscale input images.

Specifically, it generates color images in the YUV channels space when given input grayscale images in the Y channel.

Implemented a modified version of VGG-16 along with use of its Hypercolumns to concatenate with upscaling Hypercolumns (inspired from residual skip connections in ResNet) to generate UV channel output.

Trained and tested on cloud with Places dataset using Huber Loss function and Adam Optimizer.

One of the issue is that it sometimes produces sepia toned images. This is because of the simple loss function
## Dataset
Places dataset. [More details here](http://places.csail.mit.edu/)

# Results
Here are some of the best, average and worst results

## Some of the best generated colorized image

![original](images/original_good/0ace9ac43c13bf3acbab51f115282c37.jpg "<== Original Image")<== Original Image
![generated](images/generated_good/100_91.jpg "<== Generated Image")<== Generated Image

![original](images/original_good/0ac866355180c83f9fcea06703d3f5da.jpg "<== Original Image")<== Original Image
![generated](images/generated_good/100_90.jpg "<== Generated Image")<== Generated Image

![original](images/original_good/0ab85079e2e4d32a29ae8e081f35d21a.jpg "<== Original Image")<== Original Image
![generated](images/generated_good/100_85.jpg "<== Generated Image")<== Generated Image

## Some of the medium quality generated colorized image

![generated](images/generated_medium/38_529.jpg "<== Generated Image")<== Generated Image

![generated](images/generated_medium/38_538.jpg "<== Generated Image")<== Generated Image

![generated](images/generated_medium/38_554.jpg "<== Generated Image")<== Generated Image

![generated](images/generated_medium/38_560.jpg "<== Generated Image")<== Generated Image


## TODO
1. Implement a better and complex loss function
2. Provide more sample images for input, ground truth and predicted
