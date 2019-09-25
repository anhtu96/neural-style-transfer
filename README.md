# Neural style transfer
This is an implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, et al.. All the deep learning networks in this repository are done using PyTorch framework.

The concept of the paper is to combine the content of one image and the style of another image, in order to generate a new image which has the same content as the first image, whereas having the style of the latter. Here're some examples of this concept:

### Example 1
Content image: <b>Louvre</b>, Style image: [<b>The Great Wave off Kanagawa</b>](https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa) by <b>Hokusai</b>

<div align="center">
  <img src="https://raw.githubusercontent.com/anhtu96/neural-style-transfer/master/images/louvre.jpg" height="224px">
  <img src="https://raw.githubusercontent.com/anhtu96/neural-style-transfer/master/images/blue_wave.jpg" height="224px">
  <img src="https://raw.githubusercontent.com/anhtu96/neural-style-transfer/master/images/output/louvre_wave.jpg" height="400px">
</div>

### Example 2
Content image: <b>Shanghai</b>, Style image: [<b>The Starry Night</b>](https://en.wikipedia.org/wiki/The_Starry_Night) by <b>Vincent van Gogh</b>

<div align="center">
  <img src="https://raw.githubusercontent.com/anhtu96/neural-style-transfer/master/images/shanghai.jpg" height="224px">
  <img src="https://raw.githubusercontent.com/anhtu96/neural-style-transfer/master/images/starry-night.jpg" height="224px" width="300px">
  <img src="https://raw.githubusercontent.com/anhtu96/neural-style-transfer/master/images/output/shanghai_starry.jpg" height="400px">
</div>

### Model
Following the original paper, I use a pretrained VGG19 model to extract image features. I use the output features from layers `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv4_2`, `conv5_1`.

### Implementation
The goal of this problem is to minimize the loss function, which includes the content loss between final image and content image and the style loss between final image and style image. To simplify: <b>`total_loss = alpha*content_loss + beta*style_loss`</b>, in which alpha and beta are 2 parameters we can choose. A big alpha tends to minimize content loss more, whereas a big beta will help minimize style loss more.

The content loss is computed by extracting final image's features and content's features at layer `conv4_2`. It is simply computed by getting the total sum of all pixels' differences between 2 extracted features.

The style loss is more complicated, it is computed by summing the style loss at `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`. To compute style loss for a single layer, extract the style image's features and final image's features at each layer, then compute Gram matrices of these features and calculate the sum of differences between these Gram matrices. Each layer has its own style loss weight, we can choose these parameters. In summary:

<b>`style_loss = w1*loss_conv1_1 + w2*loss_conv2_1 + w3*loss_conv3_1 + w4*loss_conv4_1 + w5*loss_conv5_1`</b>

### Optimization
In this repository, I use Adam to optimize final image's parameters. There're still a lot of different optimizers to try.
