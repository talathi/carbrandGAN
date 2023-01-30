# carbrandGAN
Gan model to synthesize car images

## Data
1. Data: https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands
2. Data Preprocessing: Manual inspection of images per folder and removing false images such as Jaguar the animal; multiple cars in a single image; car logo etc..
  a. Original Dataset has 4579 images; Outlier images removed: 915 
  b. Total Images available for Training: 3664
  c. Average Image Resolution: 175 x 280 (aspect ration: 1.6)
3. Variable image resolution; Resize: 128 x 224 (aspect ratio: 1.75)
## Model Training:
1. Hinge Loss  + Gradient Pentaly + EMA smoothing + Conditional 
2. Progressive Training with Latent Dimension Modulation coupled with Differential Augmentation
3. Dual Learning Rate: lrG = 0.0001; lrD=0.0004
4. Adam Optimizer
5. Trained for 400 Epochs: 200 with latentDim: 100; 100 with latentDim: 200 and 100 with latentDim: 300
## Network
1. Fully convolutional (like DCGAN)
2. With Spectral Norm and Mish Nonlinearity
## Results

## Potential Improvements
1. Train for a longer time (may be 300-500 epochs per latent dimension size or longer)
2. Consider further cleaning of data to remove front facing and back facing images
3. Modify network architecture to incorporate style elements --stylegan or later iterations, possibly starting with pre-trained version of these
4. Modify network architecture to incorporate self attention-- SAGAN type network architecture
5. Add cosine similarity loss to further boost image variability
6. New direction could be to possibly explore stable-diffusion framework to synthesize images from text prompts using pre-trained models from huggingface




