# Image Style Transfer
Neural style transfer combines a style reference image (such as a piece of art by a well-known painter) and a content image using an optimization approach as to create an output image that resembles the content image but has been "painted" in the manner of the style reference image.

<p align="center">
<img src="https://user-images.githubusercontent.com/82305551/215335649-9cdac773-eae3-4830-b924-e94e95930c90.jpg" width="600">
</p>

This is done by optimising the output image to match the style reference image's and the content image's statistics for both content and style. A convolutional network is used to extract these data from the images.

<p align="center">
<img src="https://user-images.githubusercontent.com/82305551/215337296-462ee136-bc70-4d1f-8d98-989afc9b6861.jpg" width="600">
</p>

## Web App

This flask web app lets the user select from 4 different styles and apply them on an uploaded content image. This style transfer might take a few minutes, and is based on [this](https://arxiv.org/abs/1508.06576) paper. To start the application, run ```python main.py``` after installing pytorch, flask and pillow in the environment.

<p align="center">
<img src="https://user-images.githubusercontent.com/82305551/215336892-1b0e272a-78c1-48fb-b2af-b7d10f90a56c.png" width="600">
</p>

You can now select a style and upload an image of your choice. Processing might take some time.

<p align="center">
<img src="https://user-images.githubusercontent.com/82305551/215337026-f9958a7a-c326-41bc-8d8f-9373e00d418b.png" width="600">
</p>
