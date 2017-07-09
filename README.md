# *Image-Colorization* via Generative Adversarial Networks

<img src="data/cover.jpg" height="480px" width="640px" align="right">

This is a PyTorch implementation of *[Unsupervised Diverse Colorization via
Generative Adversarial Networks](https://arxiv.org/pdf/1702.06674.pdf)* which makes use of [batch normalization](https://arxiv.org/pdf/1502.03167v3.pdf) and [conditional concatenation](https://web.eecs.umich.edu/~honglak/icml2016-crelu-full.pdf) to improve upon [other published implementations](http://cs231n.stanford.edu/reports/2016/pdfs/224_Report.pdf) in terms of training efficiency and evaluation results.

A pair of neural networks are trained in an alternating manner to learn a conditional mapping from a single channel grayscale image to a three-channel colorized image. 
The [HSV cylindrical-coordinate encoding](https://en.wikipedia.org/wiki/HSL_and_HSV) is used rather than [RGB](https://en.wikipedia.org/wiki/RGB_color_model) as it only requires the networks to learn 2 other channels rather than three.
The mapping learned thus entails learning the *hue* and *saturation* channels given the *value* channel.

#### Implementation Architecture

A pair of [generative](https://en.wikipedia.org/wiki/Generative_model) and [discriminative](https://en.wikipedia.org/wiki/Discriminative_model) models are trained end to end. 
The generator is given uniform noise as input and is conditioned on the value channel of some given image. This conditioning gives rise ot the [conditional GAN](https://arxiv.org/pdf/1411.1784.pdf) architecture.
Unlike former image colorizer proposals, this generator is conditioned on the input at each convolutional block. This vastly improves the training procedure.

