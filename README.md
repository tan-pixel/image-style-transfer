# Image Style Transfer
Neural style transfer combines a style reference image (such as a piece of art by a well-known painter) and a content image using an optimization approach as to create an output image that resembles the content image but has been "painted" in the manner of the style reference image.

This is done by optimising the output image to match the style reference image's and the content image's statistics for both content and style. A convolutional network is used to extract these data from the images.

![result_3 (1)](https://user-images.githubusercontent.com/82305551/215332568-38dd9966-90ad-4caf-8e8f-9ac88048c34e.jpg)

This flask web app lets the user select from 4 different styles and apply them on an uploaded content image. This style transfer might take a few minutes, and is based on [this](https://arxiv.org/abs/1508.06576) paper.
