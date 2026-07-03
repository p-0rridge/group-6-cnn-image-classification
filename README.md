# Animals-10 Image Classification with CNN

## Want to test our model?
Go to: https://huggingface.co/spaces/p-0rridge/animals-10-classifier

## 2. About Project
This project explores image classification with Convolutional Neural Networks using the Animals-10 dataset. We trained and compared several CNN models, starting from a simple baseline and gradually testing deeper, wider, and transfer learning models.

The goal is to classify animal images into 10 categories and understand how different model architectures affect performance.

   ![alt text](image/image.png)

   Try it! (https://huggingface.co/spaces/p-0rridge/animals-10-classifier)

## 3. Problem statement – dataset used, task, goal
- Input: Animal images from the Animals-10 dataset
- Output: One of 10 animal classes
- Goal: Train and compare multiple CNN-based models
- Evaluation focus: Accuracy, precision, recall, loss curves, confusion matrix, and example errors
- Challenge: Images vary in size, background, lighting, pose, and visual complexity
## 4. Dataset
The dataset used in this project is the Animals-10 dataset from Kaggle.
- Source: Kaggle Animals-10 dataset
- Size: 26K medium quality animal images 
- Number of classes: 
  - `cane` – dog
  - `cavallo` – horse
  - `elefante` – elephant
  - `farfalla` – butterfly
  - `gallina` – chicken
  - `gatto` – cat
  - `mucca` – cow
  - `pecora` – sheep
  - `ragno` – spider
  - `scoiattolo` – squirrel
- Dataset link: https://www.kaggle.com/datasets/alessiocorrado99/animals10/data
- Size: [TODO: Add total number of images, e.g. around 26k images]
- License: GPL-2.0
## 5. Model architecture
## Model Architecture Comparison

| Model | Architecture |
|---|---|
| Baseline Model | 2 Conv layers + 1 MaxPooling + 1 Dropout |
| Model 2_deeper_cnn | 5 Conv layers + 3 MaxPooling + 3 Dropout<br>32×2 + 64×2 + 64 |
| Model 3_wider_cnn | 5 Conv layers + 3 MaxPooling + 3 Dropout<br>64×2 + 128×2 + 128 |
| Model 4_global_avg_pool | Baseline model + Global Average Pooling |
| Model 5_global_avg_pool_wider | Wider CNN + Global Average Pooling |
| Model 6_lr | Baseline model<br>2 Conv layers + 1 MaxPooling + 1 Dropout |
| Model 7_Augmentation | Baseline model + Data Augmentation |
| Model 8_Augmentation_wider | Wider CNN + Data Augmentation<br>5 Conv layers + 3 MaxPooling + 3 Dropout |
| Model 9_Transfer | MobileNetV2 + Dense + Dropout + Dense<br>100 frozen layers + 54 trainable layers |
| Model 10_Transfer + finetuning | MobileNetV2 + Dense + Dropout + Dense<br>90 frozen layers + 64 trainable layers |

## Major Architecture description:
### Model 1: Baseline model
A simple from-scratch CNN used as the baseline model, consisting of 2 Conv2D layers with filter sizes 32 -> 64, MaxPooling layers, Dropout for regularization, followed by Flatten and a Softmax Dense output layer.
### Model 3: Custom Deeper and Wider CNN
A from-scratch Convolutional Neural Network consisting of 5 Conv2D layers (with increased filter sizes: 64 -> 128 -> 128) interleaved with 3 Maxpooling layers and Dropout (0.2 to 0.5) to combat overfitting, followed by a Flatten and Dense layer.
#### Model 10: Transfer Learning & Fine-Tuning (Overall Winner)
Built upon Google's pre-trained **MobileNetV2** as a powerful feature extractor, topped with a custom classification head (Dense + Dropout). After initial training, advanced Fine-Tuning was applied by unfreezing and adapting the top layers (90+64 configuration) specifically to our animal dataset.

## 6. Results 
## Model Comparison

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Model 1_baseline | 59.13% | 56.13% | 54.86% |
| Model 2_deeper_cnn | 63.64% | 60.41% | 59.88% |
| Model 3_wider_cnn | 67.34% | 64.23% | 62.61% |
| Model 4_global_avg_pool | 32.28% | 23.85% | 24.64% |
| Model 5_global_avg_pool_wider | 60.70% | 56.23% | 53.56% |
| Model 6_lr 0.00005 | 58.82% | 55.69% | 54.35% |
| Model 7_Augmentation | 39.15% | 47.36% | 35.32% |
| Model 8_Augmentation_wider | 18.41% | 1.84% | 10.00% |
| Model 9_Transfer | 94.39% | 94.01% | 93.52% |
| Model 10_Transfer + finetuning | 94.88% | 94.54% | 94.14% |

## Description:
### Model 1: Baseline model
- **Performance:** Used as the starting benchmark, achieving **59.13% Test Accuracy** (56.13% Precision / 54.86% Recall).
- **Efficiency:** With only a simple CNN structure, it trained relatively fast and helped us understand the basic performance level before testing more complex models.
### Model 3: Custom Deeper and Wider CNN
- **Performance:** Achieved the highest accuracy among all self-built models with **67.34% Test Accuracy** (64.23% Precision / 62.61% Recall).
- **Efficiency:** Balanced performance well, requiring a total training time of **19 minutes** to achieve its peak accuracy.
### Model 10: Transfer Learning & Fine-Tuning (Overall Winner)
- **Performance:** The absolute benchmark champion of this project, skyrocketing to an outstanding **94.88% Test Accuracy** (94.54% Precision / 94.14% Recall).
- **Efficiency:** Thanks to the leverage of pre-trained weights, it converged in just **16 minutes** of total training time, proving to be both incredibly accurate and highly resource-efficient. Our overall candidate for deployment.

### Confusion Matrices Comparison

| Baseline CNN | Deeper+Wider CNN | Transfer Learning |
|---|---|---|
| <img src="image/image-3.png" width="300"/> | <img src="image/image-4.png" width="300"/> | <img src="image/image-5.png" width="300"/> |

## 7. Setup & installation – how to clone, install deps (requirements.txt), env setup
    pip install -r requirements.txt

## 8. Project structure
    w3_group-6-cnn-image-classification/
    ├── notebooks/          # Data preprocessing and model training notebooks
    ├── saved_models/       # Trained Keras models
    ├── app/                # Simple prediction app
    ├── data/               # Saved train/validation/test split
    ├── image/              # README images, plots, confusion matrices
    ├── results/            # Evaluation outputs
    ├── spreadsheet/        # Model comparison tables
    ├── utils.py            # Shared data loading and preprocessing functions
    ├── requirements.txt
    └── README.md

## 9. Tech stack
    - Python
    - TensorFlow / Keras
    - NumPy
    - Matplotlib
    - scikit-learn
    - KaggleHub
    - Google Colab

## 10. Run in Google Colab

This project was developed using Google Colab. To run the notebooks:

Open the notebook in Google Colab
Connect Google Drive
Load the dataset using the provided utility functions
Run the model training cells

## 11. Author/contact 
- Shan Wang - https://github.com/shaniewill
- Ursula Demling - https://github.com/p-0rridge
