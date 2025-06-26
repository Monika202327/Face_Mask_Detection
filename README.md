# Face_Mask_Detection
This project implements a deep learning model to detect whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN). It’s useful for real-time video surveillance and safety systems in public areas.

📁 Files
Face_Mask_Detection.ipynb: The main Jupyter Notebook containing the code for loading data, training the model, and performing prediction on images or video.

# Features
Image classification into two categories: With Mask and Without Mask  
CNN-based architecture built using TensorFlow/Keras  
Real-time detection using OpenCV (optional)   
Image data augmentation and preprocessing   
Evaluation metrics and visualization of accuracy/loss

# Technologies Used   
   Python  
   TensorFlow / Keras  
   OpenCV  
   matplotlib, numpy, os, seaborn   
   scikit-learn for evaluation metrics

# Dataset  
Commonly used datasets for this kind of project include:  
Face Mask Detection Dataset
You can use your own dataset or collect images with and without masks in two separate folders.

 
 # How to Run
  1.Clone the repository:   
 git clone https://github.com/yourusername/face-mask-detection.git
 cd face-mask-detection   
 2.Install dependencies:   
 pip install -r requirements.txt  
 3.Launch the Jupyter Notebook:   
 jupyter notebook Face_Mask_Detection.ipynb   
 4.Train the model or load a pre-trained model to perform    prediction.

# Evaluation  
Model performance is evaluated using:  
Accuracy  
Precision, Recall, F1-score  
Confusion Matrix  
Plots for training vs validation accuracy/loss are included.  
