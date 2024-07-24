The Enhanced Lip Reading project is an Automated Lip Reading (ALR) model that utilizes Temporal Convolutional Neural Networks (TCNs) and ResNet18. The system is trained on the Lip Reading in the Wild (LRW) dataset and can recognize words from video input. It includes a web application interface built with Django to facilitate real-time or pre-recorded video analysis.

Working Process Divided into Simpler Problems:
Data Preprocessing: Preparing and cleaning video data for model training.
Model Training: Utilizing TCNs and ResNet18 to train models on the LRW dataset.
Model Deployment: Integrating trained models into a Django web application.
User Interface: Creating a web interface to input videos and display predictions.
Real-time Processing: Enabling real-time lip reading from webcam input.


TCN (Temporal Convolutional Network)
TCNs are specialized types of Convolutional Neural Networks (CNNs) designed for sequential data. They use causal convolutions to ensure the model only considers past inputs to make predictions, preserving temporal order. They are effective for tasks involving time series, audio processing, and video analysis.

ResNet18
ResNet18 is a deep neural network architecture part of the ResNet (Residual Network) family. It consists of 18 layers with residual blocks that help in training very deep networks by allowing gradients to flow through the network more effectively. This makes it suitable for image recognition and classification tasks.

Applications in Lip Reading
TCN: Handles the temporal aspects of video data, processing sequences of frames to capture the movement of lips over time.
ResNet18: Analyzes individual frames to extract features that contribute to recognizing lip movements and forming words.
