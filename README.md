### VIOLENCE–NONVIOLENCE DETECTION

## ABOUT THE PROJECT

This project aims to classify video scenes as VIOLENCE or NON-VIOLENCE using Deep Learning.
Two models were used — a pretrained MobileNet and a custom-built ResNet-18 — to detect violent actions in frames extracted from videos.
The project demonstrates how deep learning architectures can be applied for scene understanding and human behavior recognition in real-world conditions.


### DATASET INFORMATION

Dataset Title: Violence Recognition from Videos

Citation:
M. Soliman, M. Kamal, M. Nashed, Y. Mostafa, B. Chawky, D. Khattab,
“Violence Recognition from Videos using Deep Learning Techniques”,
Proc. 9th International Conference on Intelligent Computing and Information Systems (ICICIS’19), Cairo, pp. 79–84, 2019.

Dataset Details:
- 1000 Violence videos
- 1000 Non-Violence videos
- Collected from YouTube
- Violence videos include real street fight situations under various environments and lighting conditions


### PREPROCESSING AND DATA AUGMENTATION

The frames extracted from videos are resized to 224x224 and normalized.
Data augmentation techniques used include:
- Rotation
- Zoom
- Brightness variation
- Width/Height shift
- Shear transformation
- Horizontal flip
These techniques improve generalization and help the model adapt to diverse environments.

### MODEL 1: MOBILENET (PRETRAINED)

A pretrained MobileNet model was fine-tuned for binary classification (Violence vs Non-Violence).
MobileNet is lightweight, has fewer parameters, and provides faster inference.

## Training Results:
Training Accuracy improved from 82.1% to 93.9% (Epoch 1 to 20)
Validation Accuracy stabilized around 93.3%
Loss curves showed smooth convergence without overfitting

# Observation:
MobileNet achieved excellent accuracy but required longer training time compared to ResNet-18.



### MODEL 2: CUSTOM RESNET-18 (BUILT FROM SCRATCH)
ResNet-18 was implemented from scratch using TensorFlow and Keras to explore residual learning.

## Architecture Details:

Convolutional + BatchNormalization + ReLU layers
Residual (Identity and Convolutional) Blocks
Global Average Pooling
Dropout (0.4)
Dense Softmax output layer for binary classification

## Training Configuration:

Optimizer: Adam (learning rate = 1e-4)
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 20
Batch Size: 32

Callbacks Used:
EarlyStopping – stops when validation loss stops improving
ReduceLROnPlateau – reduces learning rate when progress stagnates
ModelCheckpoint – saves the best model as “best_resnet_se_model.keras”

## Observation:
ResNet-18 trained faster than MobileNet but produced a larger model file.
It maintained high accuracy and showed stable training and validation performance.


### RESULTS SUMMARY

## Model Comparison:

Model | Training Accuracy | Validation Accuracy | Training Time | Model Size
MobileNet | 93.9% | 93.3% | Longer | Smaller
ResNet-18 | High (comparable) | High (comparable) | Faster | Larger
