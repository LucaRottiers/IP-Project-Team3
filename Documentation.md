# Guide to create a live object detection system

*Author: Luca Rottiers*.

This guide will provide you with the steps to create a live object detection system using:

- Tensorflow Object Detection API
- Raspberry PI 4B
- USB webcam

Before starting, make sure you OS is up to date by running: `sudo apt-get update` in the terminal.

## Installing Python

Most Raspberry Pies (using Raspberry Pi OS) arrive with Python preinstalled. For this project you will need to have **Python version 3.7** or above. To check your Python version, simply open a Terminal window and execute the following command:

```console
  pi@raspberrypi:~ $ python3 --version
```

## Installing OpenCV

We'll use openCV to let the python scripts control our usb webcam and configure it correctly.

### Install prerequisites

For some versions of Raspberry Pi OS we may need to install some additional packages. First make sure apt-get is fully up-to-date and install the prerequisite packages using:

```console
  pi@raspberrypi:~ $ sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 python3-dev -y
```

### Install OpenCV with pip

Finally, we can enter install OpenCV very simply using `pip`. Note that if you still have python2.7 on your system and you are not working with a virtual environment with python3, you will need to type in `pip3` rather than `pip`:

```console
  pip install opencv-contrib-python
```

### Testing

Now let’s just make sure that OpenCV is working. Open a terminal window and enter python3 to start Python. Now to make sure you have installed OpenCV correctly enter:

```console
  import cv2
  cv2.__version__
```

The result should look like:

```console
  $ python3
  Python 3.9.2 (default, Feb 24 2022, 04:06:34)
  [GCC 10.2.1 20210110] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import cv2
  >>> cv2.__version__
  '4.5.5'
```

## Setting up a virtual environment

We will configure this object detection system on a virtual environment. Run the following commands to upgrade `pip` and install the virtual environment.

```console
  pi@raspberrypi:~ $ python3 -m pip install --user --upgrade pip
  pi@raspberrypi:~ $ python3 -m pip install --user virtualenv
```

Next, we'll create a Python virtual environment for the TFLite samples (this step is optional but strongly recommended). The name of the environment will be **tflite**.

```console
  pi@raspberrypi:~ $ python3 -m venv ~/tflite
```

Whenever you want to activate the virtual environment, open a new terminal window and run this command:

```console
  pi@raspberrypi:~ $ source ~/tflite/bin/activate
```

## API

For this proof of concept we will be using the Tensorflow object detection API. Clone the Github repo with the API to your Raspberry Pi and navigate to the following directory.

```console
  pi@raspberrypi:~ $ git clone https://github.com/tensorflow/examples.git
  pi@raspberrypi:~ $ cd examples/lite/examples/object_detection/raspberry_pi
```

Afterwards we are going to run `setup.sh` to configure the API for our use case.

```console
  pi@raspberrypi:~/examples/lite/examples/object_detection/raspberry_pi $ sh setup.sh
```

Run the object detection example.

```console
  pi@raspberrypi:~/examples/lite/examples/object_detection/raspberry_pi $ python detect.py
```

This Python script is built-in to the API and is useful for detecting some standard items such as an apple, people, a laptop, ... However this offers a very limited amount of objects. Most of these objects don't fit in to our usecase or are simple not relevant for us.

In the following section we will create a custom data set and feed it to the API. This will result in the object detection API recognizing custom objects that people use during construction.

## Creating a custom dataset

For the purpose of this proof of concept, we will make a dataset with only 2 objects. Note that this dataset is fully scalable and can be used with much more objects.

Our objects include:

| Earmuffs                                | A tapemeasure                           |
|:---------------------------------------:|:---------------------------------------:|
|![earmuffs](/img/IMG20220224125312.jpg)  |![earmuffs](/img/IMG20220224125122.jpg)  |

We took 60 pictures of these objects in different lighting conditions and angles.

Using a tool called [LabelImg](https://github.com/tzutalin/labelImg) we drew borderboxes around the objects and labeled them accordingly. Each label is saved in an additional `xml` file, as you can see in the dataset folder.

Next we'll divide our data set in 2 subfolders, one called `train` with 10 pictures/xmls and another one called `validate` with 50 pictures/xmls. `train` will be used to train the machine learning algorithm and `validate` will be used to test and verify images the algorithm has never seen before.

## Training the dataset

The next step in the proces is to train our custom dataset. This will be achieved using **Google Colab**. Colaboratory, or “Colab” for short, is a product from Google Research. Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education.

The custom notebook used in this demo is also available in the Github Repo.

### Model Maker Object Detection for Construction Equipment

The Model Maker library uses transfer learning to simplify the process of training a TensorFlow Lite model using a custom dataset. Retraining a TensorFlow Lite model with your own custom dataset reduces the amount of training data required and will shorten the training time.

#### Preparation

Install the required packages:

```python
  !pip install -q tflite-model-maker
  !pip install -q tflite-support
```

```console
     |████████████████████████████████| 616 kB 5.3 MB/s 
     |████████████████████████████████| 840 kB 40.5 MB/s 
     |████████████████████████████████| 87 kB 6.3 MB/s 
     |████████████████████████████████| 3.4 MB 40.9 MB/s 
     |████████████████████████████████| 234 kB 47.3 MB/s 
     |████████████████████████████████| 596 kB 37.9 MB/s 
     |████████████████████████████████| 77 kB 5.4 MB/s 
     |████████████████████████████████| 1.1 MB 46.7 MB/s 
     |████████████████████████████████| 1.1 MB 40.4 MB/s 
     |████████████████████████████████| 1.2 MB 39.1 MB/s 
     |████████████████████████████████| 120 kB 46.6 MB/s 
     |████████████████████████████████| 6.4 MB 34.5 MB/s 
     |████████████████████████████████| 25.3 MB 1.6 MB/s 
     |████████████████████████████████| 352 kB 49.8 MB/s 
     |████████████████████████████████| 47.7 MB 73 kB/s 
     |████████████████████████████████| 99 kB 8.5 MB/s 
     |████████████████████████████████| 462 kB 45.8 MB/s 
     |████████████████████████████████| 211 kB 48.5 MB/s 
  Building wheel for fire (setup.py) ... done
  Building wheel for py-cpuinfo (setup.py) ... done
```

Import required packages:

```python
  import numpy as np
  import os

  from tflite_model_maker.config import ExportFormat, QuantizationConfig
  from tflite_model_maker import model_spec
  from tflite_model_maker import object_detector

  from tflite_support import metadata

  import tensorflow as tf
  assert tf.__version__.startswith('2')

  tf.get_logger().setLevel('ERROR')
  from absl import logging
  logging.set_verbosity(logging.ERROR)
```

Simply upload the dataset using google drive or download it and upload it to colab via the upload files option in the menu.

```python
  !unzip -q dataset.zip
```

#### Train the object detection model

Step 1: Load the dataset

- Images in train_data is used to train the custom object detection model.
- Images in val_data is used to check if the model can generalize well to new images that it hasn't seen before.

```python
  train_data = object_detector.DataLoader.from_pascal_voc(
      'Dataset/train',
      'Dataset/train',
      ['headphones', 'tapemeasure']
  )

  val_data = object_detector.DataLoader.from_pascal_voc(
      'Dataset/validate',
      'Dataset/validate',
      ['headphones', 'tapemeasure']
  )
```

Step 2: Select a model architecture

`EfficientDet-Lite[0-4]` are a family of mobile/IoT-friendly object detection models derived from the EfficientDet architecture.

In this notebook, we use EfficientDet-Lite0 to train our model. You can choose other model architectures depending on whether speed or accuracy is more important to you.

```python
  spec = model_spec.get('efficientdet_lite0')
```

Step 3: Train the TensorFlow model with the training data.

- Set epochs = 20, which means it will go through the training dataset 20 times. You can look at the validation accuracy during training and stop when you see validation loss (val_loss) stop decreasing to avoid overfitting.
- Set batch_size = 4 here so you will see that it takes 13 steps to go through the 50 images in the training dataset.
- Set train_whole_model=True to fine-tune the whole model instead of just training the head layer to improve accuracy. The trade-off is that it may take longer to train the model.

```python
  model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)
```

```console
  Epoch 1/20
  12/12 [==============================] - 66s 1s/step - det_loss: 1.7615 - cls_loss: 1.1400 - box_loss: 0.0124 - reg_l2_loss: 0.0630 - loss: 1.8245 - learning_rate: 0.0065 - gradient_norm: 1.6812 - val_det_loss: 1.7110 - val_cls_loss: 1.1310 - val_box_loss: 0.0116 - val_reg_l2_loss: 0.0630 - val_loss: 1.7740
  Epoch 2/20
  12/12 [==============================] - 14s 1s/step - det_loss: 1.6365 - cls_loss: 1.1021 - box_loss: 0.0107 - reg_l2_loss: 0.0630 - loss: 1.6995 - learning_rate: 0.0049 - gradient_norm: 1.4360 - val_det_loss: 1.5781 - val_cls_loss: 1.0874 - val_box_loss: 0.0098 - val_reg_l2_loss: 0.0630 - val_loss: 1.6411
  Epoch 3/20
  12/12 [==============================] - 13s 1s/step - det_loss: 1.5421 - cls_loss: 1.0516 - box_loss: 0.0098 - reg_l2_loss: 0.0630 - loss: 1.6050 - learning_rate: 0.0048 - gradient_norm: 1.9848 - val_det_loss: 1.4315 - val_cls_loss: 1.0132 - val_box_loss: 0.0084 - val_reg_l2_loss: 0.0630 - val_loss: 1.4944
  Epoch 4/20
  12/12 [==============================] - 14s 1s/step - det_loss: 1.4209 - cls_loss: 0.9815 - box_loss: 0.0088 - reg_l2_loss: 0.0630 - loss: 1.4839 - learning_rate: 0.0046 - gradient_norm: 2.6273 - val_det_loss: 1.2220 - val_cls_loss: 0.8530 - val_box_loss: 0.0074 - val_reg_l2_loss: 0.0630 - val_loss: 1.2849
  Epoch 5/20
  12/12 [==============================] - 20s 2s/step - det_loss: 1.2485 - cls_loss: 0.8570 - box_loss: 0.0078 - reg_l2_loss: 0.0630 - loss: 1.3115 - learning_rate: 0.0043 - gradient_norm: 2.4968 - val_det_loss: 1.0850 - val_cls_loss: 0.7708 - val_box_loss: 0.0063 - val_reg_l2_loss: 0.0630 - val_loss: 1.1480
  Epoch 6/20
  12/12 [==============================] - 13s 1s/step - det_loss: 1.0830 - cls_loss: 0.7136 - box_loss: 0.0074 - reg_l2_loss: 0.0630 - loss: 1.1460 - learning_rate: 0.0040 - gradient_norm: 3.0613 - val_det_loss: 0.9905 - val_cls_loss: 0.7201 - val_box_loss: 0.0054 - val_reg_l2_loss: 0.0630 - val_loss: 1.0535
  Epoch 7/20
  12/12 [==============================] - 15s 1s/step - det_loss: 0.9939 - cls_loss: 0.6500 - box_loss: 0.0069 - reg_l2_loss: 0.0630 - loss: 1.0568 - learning_rate: 0.0037 - gradient_norm: 3.0879 - val_det_loss: 0.8369 - val_cls_loss: 0.5773 - val_box_loss: 0.0052 - val_reg_l2_loss: 0.0630 - val_loss: 0.8999
  Epoch 8/20
  12/12 [==============================] - 13s 1s/step - det_loss: 0.8815 - cls_loss: 0.5655 - box_loss: 0.0063 - reg_l2_loss: 0.0630 - loss: 0.9445 - learning_rate: 0.0033 - gradient_norm: 3.1777 - val_det_loss: 0.7814 - val_cls_loss: 0.5369 - val_box_loss: 0.0049 - val_reg_l2_loss: 0.0630 - val_loss: 0.8444
  Epoch 9/20
  12/12 [==============================] - 13s 1s/step - det_loss: 0.7815 - cls_loss: 0.4900 - box_loss: 0.0058 - reg_l2_loss: 0.0630 - loss: 0.8445 - learning_rate: 0.0029 - gradient_norm: 2.9075 - val_det_loss: 0.7206 - val_cls_loss: 0.4833 - val_box_loss: 0.0047 - val_reg_l2_loss: 0.0630 - val_loss: 0.7836
  Epoch 10/20
  12/12 [==============================] - 16s 1s/step - det_loss: 0.7549 - cls_loss: 0.4735 - box_loss: 0.0056 - reg_l2_loss: 0.0630 - loss: 0.8179 - learning_rate: 0.0025 - gradient_norm: 3.1364 - val_det_loss: 0.7147 - val_cls_loss: 0.4821 - val_box_loss: 0.0047 - val_reg_l2_loss: 0.0630 - val_loss: 0.7777
  Epoch 11/20
  12/12 [==============================] - 14s 1s/step - det_loss: 0.7421 - cls_loss: 0.4670 - box_loss: 0.0055 - reg_l2_loss: 0.0630 - loss: 0.8051 - learning_rate: 0.0021 - gradient_norm: 3.2621 - val_det_loss: 0.6554 - val_cls_loss: 0.4384 - val_box_loss: 0.0043 - val_reg_l2_loss: 0.0630 - val_loss: 0.7184
  Epoch 12/20
  12/12 [==============================] - 13s 1s/step - det_loss: 0.6123 - cls_loss: 0.3876 - box_loss: 0.0045 - reg_l2_loss: 0.0630 - loss: 0.6753 - learning_rate: 0.0017 - gradient_norm: 2.5752 - val_det_loss: 0.6253 - val_cls_loss: 0.4227 - val_box_loss: 0.0041 - val_reg_l2_loss: 0.0630 - val_loss: 0.6883
  Epoch 13/20
  12/12 [==============================] - 14s 1s/step - det_loss: 0.6551 - cls_loss: 0.4146 - box_loss: 0.0048 - reg_l2_loss: 0.0630 - loss: 0.7181 - learning_rate: 0.0013 - gradient_norm: 3.5356 - val_det_loss: 0.5921 - val_cls_loss: 0.3860 - val_box_loss: 0.0041 - val_reg_l2_loss: 0.0630 - val_loss: 0.6551
  Epoch 14/20
  12/12 [==============================] - 14s 1s/step - det_loss: 0.6109 - cls_loss: 0.3947 - box_loss: 0.0043 - reg_l2_loss: 0.0630 - loss: 0.6739 - learning_rate: 9.6847e-04 - gradient_norm: 3.2166 - val_det_loss: 0.5890 - val_cls_loss: 0.3834 - val_box_loss: 0.0041 - val_reg_l2_loss: 0.0630 - val_loss: 0.6520
  Epoch 15/20
  12/12 [==============================] - 17s 1s/step - det_loss: 0.6028 - cls_loss: 0.3844 - box_loss: 0.0044 - reg_l2_loss: 0.0630 - loss: 0.6658 - learning_rate: 6.6478e-04 - gradient_norm: 2.8681 - val_det_loss: 0.5451 - val_cls_loss: 0.3623 - val_box_loss: 0.0037 - val_reg_l2_loss: 0.0630 - val_loss: 0.6081
  Epoch 16/20
  12/12 [==============================] - 14s 1s/step - det_loss: 0.5689 - cls_loss: 0.3713 - box_loss: 0.0040 - reg_l2_loss: 0.0630 - loss: 0.6319 - learning_rate: 4.1114e-04 - gradient_norm: 2.7948 - val_det_loss: 0.5194 - val_cls_loss: 0.3378 - val_box_loss: 0.0036 - val_reg_l2_loss: 0.0630 - val_loss: 0.5824
  Epoch 17/20
  12/12 [==============================] - 14s 1s/step - det_loss: 0.6860 - cls_loss: 0.4439 - box_loss: 0.0048 - reg_l2_loss: 0.0630 - loss: 0.7490 - learning_rate: 2.1449e-04 - gradient_norm: 3.7885 - val_det_loss: 0.5002 - val_cls_loss: 0.3211 - val_box_loss: 0.0036 - val_reg_l2_loss: 0.0630 - val_loss: 0.5632
  Epoch 18/20
  12/12 [==============================] - 13s 1s/step - det_loss: 0.5529 - cls_loss: 0.3518 - box_loss: 0.0040 - reg_l2_loss: 0.0630 - loss: 0.6159 - learning_rate: 8.0173e-05 - gradient_norm: 2.9450 - val_det_loss: 0.4948 - val_cls_loss: 0.3186 - val_box_loss: 0.0035 - val_reg_l2_loss: 0.0630 - val_loss: 0.5578
  Epoch 19/20
  12/12 [==============================] - 14s 1s/step - det_loss: 0.5991 - cls_loss: 0.3719 - box_loss: 0.0045 - reg_l2_loss: 0.0630 - loss: 0.6621 - learning_rate: 1.1867e-05 - gradient_norm: 3.0124 - val_det_loss: 0.4821 - val_cls_loss: 0.3094 - val_box_loss: 0.0035 - val_reg_l2_loss: 0.0630 - val_loss: 0.5451
  Epoch 20/20
  12/12 [==============================] - 16s 1s/step - det_loss: 0.5848 - cls_loss: 0.3773 - box_loss: 0.0042 - reg_l2_loss: 0.0630 - loss: 0.6479 - learning_rate: 1.1431e-05 - gradient_norm: 2.7479 - val_det_loss: 0.4726 - val_cls_loss: 0.3023 - val_box_loss: 0.0034 - val_reg_l2_loss: 0.0630 - val_loss: 0.5357
```

Step 4. Evaluate the model with the validation data.

After training the object detection model using the images in the training dataset, use the 10 images in the validation dataset to evaluate how the model performs against new data it has never seen before.

As the default batch size is 64, it will take 1 step to go through the 10 images in the validation dataset.

```python
  model.evaluate(val_data)
```

```console
  1/1 [==============================] - 6s 6s/step

  {'AP': 0.6970571,
  'AP50': 1.0,
  'AP75': 0.9590304,
  'AP_/headphones': 0.72148347,
  'AP_/tapemeasure': 0.67263085,
  'APl': 0.6970571,
  'APm': -1.0,
  'APs': -1.0,
  'ARl': 0.73333335,
  'ARm': -1.0,
  'ARmax1': 0.7222222,
  'ARmax10': 0.73333335,
  'ARmax100': 0.73333335,
  'ARs': -1.0}
```

Step 5: Export as a TensorFlow Lite model.

Export the trained object detection model to the TensorFlow Lite format by specifying which folder you want to export the quantized model to. The default post-training quantization technique is full integer quantization.

```python
  model.export(export_dir='.', tflite_filename='construction.tflite')
```

Step 6: Evaluate the TensorFlow Lite model.

Several factors can affect the model accuracy when exporting to TFLite:

Quantization helps shrinking the model size by 4 times at the expense of some accuracy drop.

The original TensorFlow model uses per-class non-max supression (NMS) for post-processing, while the TFLite model uses global NMS that's much faster but less accurate.

Keras outputs maximum 100 detections while tflite outputs maximum 25 detections.

Therefore you'll have to evaluate the exported TFLite model and compare its accuracy with the original TensorFlow model.

```python
  model.evaluate_tflite('construction.tflite', val_data)
```

```console
  10/10 [==============================] - 28s 3s/step

  {'AP': 0.7119401,
  'AP50': 1.0,
  'AP75': 1.0,
  'AP_/headphones': 0.74714756,
  'AP_/tapemeasure': 0.67673266,
  'APl': 0.7119401,
  'APm': -1.0,
  'APs': -1.0,
  'ARl': 0.7395833,
  'ARm': -1.0,
  'ARmax1': 0.7395833,
  'ARmax10': 0.7395833,
  'ARmax100': 0.7395833,
  'ARs': -1.0}
```

We can see that our model has an accuracy of **71.19%** which is enough for an edge device.

```python
  # Download the TFLite model to your local computer.
  from google.colab import files
  files.download('construction.tflite')
```

## Deploying the trained model on Raspberry Pi

Transfer the `construction.tflite` to the raspberry pi.

Op a terminal window, activate the virtual environment like in the beginning and navigate to the directory: *examples/lite/examples/object_detection/raspberry_pi*.

Copy the `construction.tflite` to the current directory using the following command:

```console
  pi@raspberrypi:~/examples/lite/examples/object_detection/raspberry_pi $ cp ~/Downloads/construction.tflite .
```

Now run the `detect.py` script but we'll make sure it uses our custom model instead of the default one. To achieve this, we'll run:

```console
  pi@raspberrypi:~/examples/lite/examples/object_detection/raspberry_pi $ python detect.py --model construction.tflite
```

The result is a live feed from the usb webcam where our model detects a pair of earmuffs and a tapemeasure.
