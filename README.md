# AiCoinXpert

## Description

AI project to detect valuable coins in a video frame.

Will be implemented in Python the following steps:

- Collect dataset with their pictures of Euros coins since their creation in 2002 to nowadays
- Analyse the dataset to extract trends and insights
- Define storage for Database and Pictures
- Train/Test models for object detection and classification
- Evaluate models
- Create API to serve the model
- Test the API with a web application
- Monitor the API

## Installation project 

If Yolo model on video does not work and you have an error like this one:
`qt.qpa.plugin: Could not load the Qt platform plugin "xcb" `

To fix the problem do the folowwing command:

- poetry remove opencv-python
- poetry add opencv-python
- sudo apt-get install libsm6
- sudo apt-get install libgl1-mesa-glx
- sudo apt-get install libglib2.0-dev

After installing the package, try running your Python code again and see if the error has been resolved.

- Important Notice:

  - If devcontainer does not start it sometimes because of the external webcam that is not found to resolve this we should unplug and replug it manually
  - If pytorch or ultralitycs does not work because of some lib.. not found in PATH it is require to install ultralitycs with the following command:

    - pip install ultralytics

  - for push to gerrit:
    (send to gerrit) - git push origin HEAD:refs/for/master
    (allow to merge on gerrit) - git push origin master
