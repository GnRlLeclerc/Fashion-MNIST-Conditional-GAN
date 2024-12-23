# Conditional GAN - Fashion MNIST

![Generator Showcase](./showcase.png)

A toy Conditional GAN trained on the Fashion-MNIST dataset for the [Deep Learning Computer Vision](https://perso.ensta-paris.fr/~franchi/Cours/IA323.html) course at ENSTA.

The dependencies needed can be found in [`requirements.txt`](./requirements.txt).

In order to train and showcase the model, run the following command:

```bash
python main.py
```

## Techniques used

- Because the GAN generates images with `float32` values ranging from -1 to 1, we convert the dataset images from `uint8` (0, 255) to this interval as well.
- In order to avoid the discriminator outperforming the generator, for each batch, we only backpropagate and train the discriminator if it cannot recognize more than 90% of the fake images as fake. This prevents the generator from outputting random noise during training because the discriminator already has a 100% detection rate.

## Project Overview

```
├── display.py        # Display utilities
├── fashion_mnist.py  # Dataset
├── main.py           # Entrypoint: train & showcase
├── networks.py       # Generator & Discriminator models
├── train_cgan.py     # Training function
└── utils.py          # Utilities
```
