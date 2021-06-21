# Handwritten-Characters-Generation-using-GANs
Handwritten Characters Generation using GANs in TensorFlow

* This is one of the cool projects I did in my internship period. It introduced me to GANs.
* The Dataset can be downloaded from [here](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format).
* GANs consists of two deep neural networks called Generator and Discriminator.
* For the Generator network, I used 2 Dense layers followed by 2 Conv2D layers.
* Applied Batch Normalisation to each of these layers.
* For the Discriminator network, the architecture is the reverse of that of the Generator network.
* A random noise is generated first with the dimensions and is fed into the Generator network.
* It's output(fake data) is then fed into the Discriminator network.
* The model is trained for 80 epochs and observed quite a lot of improvement during the training period.

![Generated Alphabets](https://user-images.githubusercontent.com/74998474/122776186-81ecbd00-d2c8-11eb-8e28-4d3cd83762c6.png)

