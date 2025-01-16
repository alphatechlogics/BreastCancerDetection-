# Breast Tumor Detection Using Deep Learning 🤖🩺

Welcome to the Breast Tumor Detection project! This project demonstrates to build, train, and deploy a deep learning model that detects breast tumors from mammogram images.

---

## 📊 Dataset Overview

- **Source:**  
  The dataset is available on [Kaggle](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection) and contains **3,383** annotated mammogram images.

- **Preprocessing:**  
  Each image underwent:

  - **Auto-orientation of pixel data:** Ensuring correct viewing angles by stripping EXIF orientation data.
  - **Resizing to 640x640 pixels:** Standardizing the image dimensions prior to further processing.

- **Usage:**  
  The dataset is ideal for:
  - Breast tumor detection and classification
  - Training deep learning models for medical imaging
  - Research in healthcare and medical diagnostics

---

## 🔄 Data Preparation

- **Image Loading & Preprocessing:**  
  The code uses TensorFlow’s `ImageDataGenerator` to load images from a folder-based structure.

  - **Rescaling:** All pixel values are normalized between 0 and 1.
  - **Resizing:** Images are downscaled to **64×64 pixels** for training efficiency.
  - **Color Conversion:** Mammograms are processed in grayscale, which reduces computational complexity while preserving crucial features.

- **Folder Structure:**  
  The images are organized into training, testing, and validation directories, facilitating a clear separation of data for model evaluation.

---

## 🧠 Model Architecture & Details

The model is an enhanced Convolutional Neural Network (CNN) designed specifically for mammogram classification. Here’s a breakdown of its components:

1. **Convolutional Blocks:**

   - **Multiple Layers:**  
     The network starts with several blocks of convolutional layers. Each block extracts increasingly complex features from the input images.
   - **Activation & Batch Normalization:**  
     After each convolution, a ReLU activation function is applied to introduce non-linearity, followed by batch normalization for faster convergence and more stable training.
   - **Pooling Layers:**  
     Max pooling layers reduce the spatial dimensions, focusing on the most salient features and reducing overfitting.
   - **Dropout:**  
     Dropout layers are interspersed to prevent overfitting by randomly disabling a fraction of the neurons during training.

2. **Fully Connected Layers:**

   - **Flattening:**  
     The output of the convolutional blocks is flattened into a one-dimensional vector.
   - **Dense Layers:**  
     A dense layer with 128 neurons further processes the features. Again, batch normalization and ReLU activation follow for improved training dynamics.
   - **Final Output Layer:**  
     A single neuron with a sigmoid activation function is used for binary classification (i.e., detecting if a tumor is present or not).

3. **Optimizer & Loss Function:**

   - **Optimizer:**  
     The Adam optimizer is chosen for its efficiency and adaptive learning rate capabilities.
   - **Loss Function:**  
     Binary cross-entropy is used, which is well-suited for binary classification tasks.

4. **Callbacks for Enhanced Training:**
   - **Early Stopping:**  
     Monitors the validation loss and stops training when performance stops improving, while restoring the best weights.
   - **Learning Rate Reduction:**  
     A scheduler reduces the learning rate when the validation loss plateaus, allowing finer tuning of the weights in later epochs.

---

## 🏋️ Training Strategy

- **Training Process:**  
  The model is trained for a maximum of 100 epochs using the training set, while the validation set is used to monitor overfitting and generalization capabilities.
- **Evaluation:**  
  The performance is tracked through metrics such as accuracy and loss. Both training and validation progress help in adjusting hyperparameters and preventing underfitting/overfitting.

- **Performance Challenges:**  
  In practice, achieving high accuracy in the 60–70% range may indicate the need for:
  - Adjustments in hyperparameters (e.g., learning rate, dropout rate)
  - Additional data augmentation to increase dataset variability
  - Potential improvements in model architecture or input resolution

---

## 💾 Saving & Deployment

- **Model Saving:**  
  Once trained, the model is saved to disk (as `breast.h5`) for future inference tasks.
- **Streamlit Deployment:**  
  A separate Streamlit application is built to load the saved model and allow real-time predictions. This interactive dashboard lets users upload a mammogram image to receive a prediction, displaying both a classification result and a prediction confidence score.

- **Caching Improvements:**  
  The Streamlit app uses the latest resource caching methods to efficiently manage model loading and display images using modern arguments (`st.cache_resource` and `use_container_width`), ensuring compatibility with the latest versions.

---

## 🚀 How to Get Started

Follow these steps to get started with the Breast Tumor Detection project:

1. **Clone the Repository:**  
   Open your terminal and clone the repository using:

```bash
git clone https://github.com/alphatechlogics/BreastCancerDetection-.git
cd BreastCancerDetection-
```

2. **Create a Virtual Environment:**
   It’s recommended to use a virtual environment to manage dependencies.

Using venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
   Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

4. **For real-time predictions, you can run the Streamlit app:**

This will start a local server where you can upload mammogram images and view the prediction results.

```bash
streamlit run app.py
```
