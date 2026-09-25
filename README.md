# Animal Face Classification Using PyTorch CNN

A beginner-friendly computer-vision project that classifies animal-face images into **cat**, **dog**, and **wild** using a Convolutional Neural Network (CNN) built with PyTorch.

This README is based on the uploaded `animal_faces.ipynb` notebook and documents the actual architecture, data flow, training setup, and reported results.

## Project overview

The notebook implements the following pipeline:

```text
AFHQ image folders
        |
        v
Collect image paths + labels
        |
        v
Pandas DataFrame
        |
        v
LabelEncoder
(cat=0, dog=1, wild=2)
        |
        v
CustomImageDataset
        |
        v
DataLoader
(batch size = 16)
        |
        v
CNN
        |
        v
CrossEntropyLoss
        |
        v
Backpropagation
        |
        v
Adam optimizer
        |
        v
Validation / Test
        |
        v
Single-image prediction
```

## Dataset

The notebook uses the **AFHQ (Animal Faces-HQ)** dataset and works with three classes:

- `cat`
- `dog`
- `wild`

The image folders are read from:

```text
/content/animal-faces/afhq
```

The notebook first collects image paths from both the AFHQ `train` and `val` directories into one DataFrame, then creates a new random split:

| Split | Images | Approx. share |
|---|---:|---:|
| Train | 11,291 | 70% |
| Validation | 2,420 | 15% |
| Test | 2,419 | 15% |

> **Important:** This is not the original AFHQ train/validation split. The notebook combines the available paths first and then performs its own 70/15/15 split.

Dataset source used by the notebook:

https://www.kaggle.com/datasets/andrewmvd/animal-faces

The notebook output identifies the dataset license as **CC BY-NC 4.0**. Check the dataset page for the current license and usage terms before redistribution.

## CNN architecture

The model is implemented in:

```python
class Net(nn.Module):
```

### Architecture

```text
Input
B x 3 x 128 x 128
        |
        v
Conv2d(3 -> 32, kernel=3, padding=1)
B x 32 x 128 x 128
        |
        v
MaxPool2d(2,2)
B x 32 x 64 x 64
        |
        v
ReLU
        |
        v
Conv2d(32 -> 64, kernel=3, padding=1)
B x 64 x 64 x 64
        |
        v
MaxPool2d(2,2)
B x 64 x 32 x 32
        |
        v
ReLU
        |
        v
Conv2d(64 -> 128, kernel=3, padding=1)
B x 128 x 32 x 32
        |
        v
MaxPool2d(2,2)
B x 128 x 16 x 16
        |
        v
ReLU
        |
        v
Flatten
B x 32,768
        |
        v
Linear(32,768 -> 128)
B x 128
        |
        v
Linear(128 -> 3)
B x 3 logits
        |
        v
cat / dog / wild
```

### Visual architecture diagram

![Exact CNN architecture](animal_faces_cnn_architecture_exact.png)

### Layer summary

| Layer | Operation | Output shape | Parameters |
|---|---|---|---:|
| Input | RGB image | `B x 3 x 128 x 128` | - |
| Conv1 | `3 -> 32`, k=3, p=1 | `B x 32 x 128 x 128` | 896 |
| MaxPool1 | k=2, s=2 | `B x 32 x 64 x 64` | 0 |
| ReLU1 | activation | `B x 32 x 64 x 64` | 0 |
| Conv2 | `32 -> 64`, k=3, p=1 | `B x 64 x 64 x 64` | 18,496 |
| MaxPool2 | k=2, s=2 | `B x 64 x 32 x 32` | 0 |
| ReLU2 | activation | `B x 64 x 32 x 32` | 0 |
| Conv3 | `64 -> 128`, k=3, p=1 | `B x 128 x 32 x 32` | 73,856 |
| MaxPool3 | k=2, s=2 | `B x 128 x 16 x 16` | 0 |
| ReLU3 | activation | `B x 128 x 16 x 16` | 0 |
| Flatten | feature maps -> vector | `B x 32,768` | 0 |
| Linear | `32,768 -> 128` | `B x 128` | 4,194,432 |
| Output | `128 -> 3` | `B x 3` | 387 |

**Total trainable parameters: 4,288,067**

## Why the dimensions change

The convolution layers use:

```python
kernel_size=3
padding=1
```

so height and width stay unchanged during convolution.

Each:

```python
MaxPool2d(2,2)
```

halves height and width:

```text
128 -> 64 -> 32 -> 16
```

The number of channels grows because the convolution layers learn more filters:

```text
3 -> 32 -> 64 -> 128
```

After the third pooling layer:

```text
128 x 16 x 16 = 32,768
```

These 32,768 values are flattened into one vector and sent to the fully connected layers.

## Data preprocessing

The notebook uses:

```python
labelencoder = LabelEncoder()
labelencoder.fit(data_df["labels"])

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])
```

So each image is:

1. resized to `128 x 128`
2. converted to a tensor
3. converted to a floating-point image type

The label mapping is:

```text
cat  -> 0
dog  -> 1
wild -> 2
```

## Custom PyTorch Dataset

The `CustomImageDataset` class connects the DataFrame to PyTorch.

Its main responsibilities are:

```text
__init__()       -> prepare the dataset
__len__()        -> tell PyTorch how many samples exist
__getitem__(idx) -> load one image and its label
```

For each requested index, it:

```text
find path
   |
open image
   |
convert to RGB
   |
apply transforms
   |
return (image, label)
```

## DataLoader

The notebook creates:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)
```

and equivalent loaders for validation and testing.

The DataLoader groups individual samples into batches so the CNN can process multiple images together.

## Training setup

```python
LR = 1e-4
BATCH_Size = 16
EPOCHS = 10

criterian = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)
```

### One training batch

The essential training cycle is:

```python
optimizer.zero_grad()

output = model(inputs.to(device))

train_loss = criterian(
    output,
    labels.to(device)
)

train_loss.backward()

optimizer.step()
```

In plain English:

```text
1. Clear old gradients
2. Run the images through the CNN
3. Calculate the loss
4. Compute gradients with backward()
5. Update the weights with Adam
```

This repeats for every batch and every epoch.

## Validation

After each training epoch, the notebook evaluates the validation set using:

```python
with torch.no_grad():
```

It then:

- calculates validation loss
- finds predicted classes using `argmax`
- counts correct predictions
- computes validation accuracy

For example:

```text
model output = [1.2, 4.8, 0.5]

argmax -> 1
```

With this project's label mapping:

```text
1 -> dog
```

## Reported training results

The notebook ran for 10 epochs and reported:

| Epoch | Train loss | Val loss | Train acc (%) | Val acc (%) |
|---:|---:|---:|---:|---:|
| 1 | 0.0596 | 0.0147 | 97.0065 | 96.2810 |
| 2 | 0.0472 | 0.0161 | 97.6353 | 96.0744 |
| 3 | 0.0318 | 0.0159 | 98.5298 | 95.9504 |
| 4 | 0.0230 | 0.0172 | 98.8664 | 96.0331 |
| 5 | 0.0206 | 0.0183 | 99.0612 | 95.6198 |
| 6 | 0.0117 | 0.0160 | 99.5129 | 96.4463 |
| 7 | 0.0107 | 0.0197 | 99.5483 | 95.8678 |
| 8 | 0.0103 | 0.0242 | 99.4686 | 95.1240 |
| 9 | 0.0046 | 0.0204 | 99.8317 | 96.2397 |
| 10 | 0.0068 | 0.0170 | 99.7077 | 96.5702 |

### Test result

The notebook reports:

```text
Test Accuracy: 97.437%
Reported Test Loss: 0.0186
```

> **Loss note:** the notebook accumulates batch losses and divides by `1000` before printing. Therefore, the displayed loss values are not the standard mean loss per batch.

## Single-image prediction

The notebook defines:

```python
def pred_image(image_path):
  image = Image.open(image_path).convert("RGB")
  image = transform(image).to(device)

  output = model(image.unsqueeze(0))

  output = torch.argmax(output, axis=1)

  return labelencoder.inverse_transform(
      output.cpu()
  )[0]
```

The flow is:

```text
image
  |
  v
RGB
  |
  v
resize + tensor transform
  |
  v
add batch dimension
  |
  v
CNN
  |
  v
3 logits
  |
  v
argmax
  |
  v
class index
  |
  v
LabelEncoder inverse transform
  |
  v
"cat" / "dog" / "wild"
```

The sample prediction in the notebook returned:

```text
dog
```

## Project structure

A simple GitHub repository can look like:

```text
animal-face-classification/
│
├── animal_faces.ipynb
├── README.md
├── animal_faces_cnn_explanation.pdf
└── animal_faces_cnn_architecture_exact.png
```

## How to run in Google Colab

1. Open `animal_faces.ipynb` in Google Colab.
2. Select a GPU runtime when available.
3. Install/import the required libraries.
4. Download/extract the AFHQ dataset.
5. Run the cells in order.
6. Train the model for 10 epochs.
7. Run the test/evaluation cell.
8. Use `pred_image("path/to/image.jpg")` for a new image.

Typical Python packages used by the notebook include:

```bash
pip install kaggle torch torchvision pandas scikit-learn matplotlib pillow tqdm torchsummary
```

## Important notes from the notebook

- The original AFHQ `train` and `val` paths are combined before the notebook creates its own train/validation/test split.
- `shuffle=True` is used for validation and test DataLoaders. Shuffling is generally unnecessary during evaluation.
- The Dataset stores encoded labels on the selected device. A more typical design is to keep dataset tensors on CPU and move labels to the device inside the training loop.
- The notebook prints loss using division by `1000`, not by the actual number of batches.
- The model is a small CNN built from scratch; it does not use transfer learning.

## Possible future improvements

- Preserve the official dataset split or use an explicit fixed random seed.
- Add a confusion matrix.
- Report precision, recall, and F1-score for each class.
- Save the trained model with `torch.save`.
- Save the label mapping with the model.
- Add data augmentation such as random flips/crops.
- Use a learning-rate scheduler.
- Compare this CNN with transfer learning models such as ResNet or EfficientNet.

## License

This repository does not define a code license in the uploaded notebook.

The dataset information shown by the notebook identifies the AFHQ dataset license as:

**CC BY-NC 4.0 (Attribution-NonCommercial)**

For redistribution or commercial use, verify the dataset's current license and conditions on the source page.
