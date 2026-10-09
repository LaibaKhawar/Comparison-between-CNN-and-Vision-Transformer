# CNN vs Vision Transformer on CIFAR-10

A comparison on CIFAR-10 between a small convolutional network trained from scratch and an ImageNet-pretrained ViT-Tiny fine-tuned with `timm`, in PyTorch.

**Case study:** [Read it](https://laiba-khawar-portfolio.vercel.app/work/cnn-vs-vit)

## How it works

All code is in `Vit_Vs_CNN(CIFAR_10).ipynb` (run on Google Colab with a GPU).

**Data (cells 3 and 4).** torchvision CIFAR-10. Training images use random crop (32, padding 4) and horizontal flip; all images are normalised with mean and std 0.5 per channel. The 10,000-image CIFAR-10 test set is passed to the training loop as the validation set and is also used for the final evaluation.

**SimpleCNN (cell 5).** Two blocks of Conv 3x3 (32 then 64 channels), BatchNorm, ReLU and 2x2 max-pooling, then Linear(4096, 256), ReLU, Dropout 0.5, Linear(256, 10). Trained with Adam (lr 0.001), batch size 128.

**Pretrained ViT-Tiny (cells 8 and 13).** `timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)`. Images are resized to 224x224 (this also replaces the training augmentation with resize plus normalisation). Fine-tuned with AdamW (lr 1e-4), batch size 64.

**Training loop (cell 7).** Up to 10 epochs, StepLR (step 5, gamma 0.5), early stopping on validation loss with patience 5 (not triggered in either run). After training, `evaluate` prints a scikit-learn classification report and plots a confusion matrix.

**From-scratch ViT (cell 6).** A ViT with 4x4 patches, 256-dim embeddings, a CLS token, learned positional embeddings, 6 pre-norm encoder blocks with 8 heads, and an MLP head is defined, but it is not trained or evaluated in the recorded outputs (the cell that would train it, cell 10, is commented out).

## Results

Final validation accuracy after 10 epochs (validation set = CIFAR-10 test set):

| Model | Accuracy | Source |
| --- | --- | --- |
| SimpleCNN (from scratch) | 69.28% | cell 12 |
| ViT-Tiny, ImageNet-pretrained, fine-tuned | 97.33% | cell 13 |

Per-class precision, recall and F1 and confusion matrices for both models are in the same cells.

The comparison is not like-for-like: the ViT starts from ImageNet pretraining and sees 224x224 upsampled images, while the CNN is trained from random initialisation at 32x32.

## Repository contents

- `Vit_Vs_CNN(CIFAR_10).ipynb`: models, training loop and evaluation.
- `requirements.txt`: jupyter, matplotlib, scikit-learn, seaborn, timm, torch, torchvision.

## Running it

The notebook was run on Google Colab with a GPU (Python 3.11 according to its outputs). Fine-tuning ViT-Tiny at 224x224 for 10 epochs is slow without a GPU.

```bash
git clone https://github.com/LaibaKhawar/Comparison-between-CNN-and-Vision-Transformer.git
cd Comparison-between-CNN-and-Vision-Transformer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook "Vit_Vs_CNN(CIFAR_10).ipynb"
```

CIFAR-10 is downloaded automatically to `./data`, and the pretrained ViT weights are downloaded from the Hugging Face Hub by `timm`.

Cell 4 ends with `classes = trainset.classes`, but the dataset variable is named `train_dataset`. Change it to `classes = train_dataset.classes` (or delete the line) before running.

## Known limitations

- **No separate validation set.** The CIFAR-10 test set is used for monitoring, early stopping and the final numbers, so there is no untouched test set.
- **The from-scratch ViT is never trained**, so the notebook does not show how a ViT without pretraining compares with the CNN.
- **Unequal setups.** Pretraining, input resolution, optimiser, batch size and augmentation all differ between the two models.
- Single run each, with no fixed random seed.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laiba-k-00b2b1249/)
