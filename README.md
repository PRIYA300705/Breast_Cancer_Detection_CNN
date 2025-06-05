# Breast Cancer Classification using CNN

![Python](https://img.shields.io/badge/Python-3.6+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/DeepLearning-TensorFlow-orange?logo=tensorflow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

## Overview

This project uses a **Convolutional Neural Network (CNN)** to classify **mammographic breast cancer images** as either:

- **PB** – Probable Benign  
- **PM** – Probable Malignant

The aim is to assist medical professionals in early detection and classification of breast cancer through deep learning.

---

## Folder Structure
```
/breast-cancer-cnn/
├── images.zip
├── labels.xlsx
├── breast_cancer_cnn.py
├── breast_cancer_cnn.ipynb
```

---

## Tech Stack

- **Python 3.6+**
- **TensorFlow / Keras** – CNN Model
- **OpenCV** – Image preprocessing
- **Pandas** – Excel data processing
- **NumPy** – Numerical operations
- **scikit-learn** – Train/test splitting

---

###  Option 1: Using Python Script (`breast_cancer_cnn.py`)

1. **Place Your Files**  
   Make sure the following are in the same directory:
   - `breast_cancer_cnn.py`
   - `labels.xlsx`
   - A folder named `images/` containing your mammographic `.jpg` files

2. **Install Dependencies**  
   Run this command to install the required libraries:
   ```bash
   pip install tensorflow opencv-python pandas numpy scikit-learn

### Option 2: Using Jupyter Notebook (`breast_cancer_cnn.ipynb`)

1. **Open the Notebook**  
   Launch Jupyter Notebook or open the notebook in Google Colab.

2. **Upload Files**  
   Ensure the following files are present in the same directory:
   - `images/` folder (extracted from `images.zip`)
   - `labels.xlsx`

3. **Run All Cells**  
   Execute the notebook cells sequentially. It will:
   - Load and preprocess image data
   - Split data into training/testing sets
   - Train a CNN model on the dataset
   - Evaluate model accuracy
   - Allow prediction on custom images

4. **Test with Custom Images**  
   In the last cell, update the image path:
   ```python
   predict_image("your_image.jpg")


---

## Prerequisites

Install the required Python libraries using:

```bash
pip install tensorflow opencv-python pandas numpy scikit-learn
```

---

### **Author**  
**Izaan Ibrahim Sayed**  
Email: izaanahmad37@gmail.com  
GitHub: [github.com/izaanahmad37](https://github.com/izaanibrahim37) 


