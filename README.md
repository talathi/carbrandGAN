# carbrandGAN
Gan model to synthesize car images

## Data
Data: https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands
Data Preprocessing: Manual inspection of images per folder and removing false images such as Jaguar the animal; multiple cars in a single image and car logo etc..
Center Crop: 256 x 158
## Model Training:
1. Hinge Loss  + Gradient Pentaly + EMA smoothing + Cosine Similarity (optional)
2. Progressive Training with Latent Dimension Modulation
3. Dual Learning Rate: lrG = 0.0001; lrD=0.0004
## Network
1. Fully convolutional (like DCGAN)
2. With Spectral Norm and Mish Nonlinearity
3. Optinal-- Choise to use self-attention

