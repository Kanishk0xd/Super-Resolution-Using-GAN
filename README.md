v# ESRGAN Super-Resolution on DIV2K Dataset

## 📌 Overview
This project implements an Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) to upscale low-resolution images to high-resolution images using deep learning. The model is trained on the DIV2K dataset and includes adversarial training with a generator and discriminator.

## 🚀 Features
- Uses ESRGAN for high-quality image super-resolution.
- Adversarial training with a generator and discriminator.
- Implements VGG perceptual loss for feature-based loss computation.
- Supports training and inference on Google Colab.
- Allows users to upload and enhance images.

## 🛠️ Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- albumentations
- Google Colab (for cloud-based training)
- VS Code (optional for local training)

### Installation
```bash
pip install torch torchvision albumentations matplotlib tqdm
```

## 📂 Dataset
This project uses the **DIV2K dataset** for training and validation. Download the dataset from:
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Extract the dataset and structure it as follows:
```
/data
 ├── DIV2K_train_HR/   # High-resolution images
 ├── DIV2K_valid_HR/   # Validation high-resolution images
```

## 🎯 Model Architecture
The project includes:
- **Generator**: A deep CNN-based network trained to upsample images.
- **Discriminator**: A CNN-based network that distinguishes real high-res images from generated ones.
- **Loss Functions**:
  - L1 Loss for pixel-wise similarity.
  - VGG Loss for perceptual similarity.
  - Adversarial Loss for realism.

## 🔧 Training
Run the training script:
```python
train(
    loader = train_loader,
    disc = model["discriminator"],
    gen = model['generator'],
    opt_gen = optimizer['generator'],
    opt_disc = optimizer['discriminator'],
    l1 = nn.L1Loss(),
    vgg_loss = VGGLoss(device),
    g_scaler = generator_scaler,
    d_scaler = discriminator_scaler,
    tb_step = 0,
    device = device,
    num_epochs = 150
)
```

## 🖼️ Image Enhancement
To upscale an image using the trained model:
```python
upload_and_enhance()
```
This function allows users to upload a low-resolution image and generate an enhanced high-resolution version.

## 📥 Saving and Loading Model
Save the trained model:
```python
torch.save(model['generator'].state_dict(), 'generator_model.pth')
torch.save(model['discriminator'].state_dict(), 'discriminator_model.pth')
```
Load the model for inference:
```python
model['generator'].load_state_dict(torch.load('generator_model.pth'))
model['discriminator'].load_state_dict(torch.load('discriminator_model.pth'))
```

## 🎉 Results
After training, the model generates high-resolution images with improved details. Use the visualization script to compare low-resolution and high-resolution images:
```python
fig, axes = plt.subplots(2, 5, figsize=(14, 7))
for i, (low_res, high_res) in enumerate(val_loader):
    if i >= 5:
        plt.show()
        break
    axes[0, i].imshow(low_res[0].permute(1, 2, 0))
    axes[0, i].set_title("Low Resolution")
    axes[1, i].imshow(fake_high_res[0].permute(1, 2, 0))
    axes[1, i].set_title("High Resolution")
```

## 🔥 Optimizations
- **Enable Mixed Precision Training**: Use `torch.cuda.amp` for faster training.
- **Increase Batch Size**: If you have more GPU memory, increase batch size.
- **Use Pretrained Models**: Start training from a pretrained ESRGAN model.

## 📜 License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Feel free to contribute by submitting pull requests or opening issues.

---
🔗 **Author:** Your Name  
📧 **Contact:** your.email@example.com  
📁 **GitHub Repository:** [GitHub Link](https://github.com/yourrepo)

