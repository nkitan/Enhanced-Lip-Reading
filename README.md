# Automatic-Lip-Reading
## Enhanced-Lip-Reading
This project contains an Automated Lip Reading (ALR) model using Temporal Convolutional Neural Networks and ResNet18. Two models have been trained for our application using the Lip Reading in the Wild (LRW) dataset.

### Based Off
This project was based off mpc001's repository Lipreading_using_Temporal_Convolutional_Networks - https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

and largely on treevesy's Linguistic-Learners-Automatic-Lip-Reading - https://github.com/treevesy/Linguistic-Learners-Automatic-Lip-Reading

### Model Directory
To use this application you will need to download the models from this Google Drive directory below
https://drive.google.com/drive/folders/1Ul-bhYOithL3cFqUdpEaYSm4bTBeLCfG?usp=sharing

### Where to Place Models
The model directory contains two models which our group trained (10 word model and 20 word model) and another model which has been trained on the full 500 words by other data scientists.
To use these models you will need to move them into the lip_reader_ai/models folder and then after starting the application you will be able to select which model you want to use on a live web-cam stream or on uploaded pre-recorded videos.

This application uses Django for the front-end to make the models that have been used accessable in a web application. 

# Requirements

1. Python (v3.0^)
2. GPU (Optional, can be run off CPU but predictions can take a long time)

# How to get running
1. Clone repository on local machine.

There are two methods to get the dependencies installed
### Using setup script
2.1 run command `chmod +x setup.sh`
2.2 run command `./setup.sh`

### Manually
2.1 run command `python -m venv py_venv` in your terminal.
2.2 run command `source py_venv/Scripts/activate`
2.3 run command `pip install -r requirements.txt`

Once the dependencies have been installed, the server can be started
3. run command `python manage.py runserver`
4. After the server is up and running open up your web browser and navigate to `http://localhost:8000` or `http://127.0.0.1:8000`
