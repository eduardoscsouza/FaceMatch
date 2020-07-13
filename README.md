# **FaceMatch - An Image Face Verifier**
Final project for **SCC0251/SCC5830 - Image Processing - 1st Semester of 2020** at University of São Paulo.



## **Authors**
 - Eduardo Santos Carlos de Souza (NUSP 9293481)
 - Guilherme Hideo Tubone (NUSP 9019403)



## **Abstract**
This project aims to build a system capable of, given two images, determine if both have the face of the same person or different people in them, known as the face verification problem. It uses image segmentation to separate the faces from the rest of the images, and image classification to determine whether the cropped faces belong to the same person. The images used are of photographic images of people, where their faces are visible, from the CelebA dataset. Possible applications revolve around security, where it is needed to verify someone's identity by an image of them, such as unlocking a phone and many surveillance systems.



## **Main Objectives**
As stated in the abstract, our objective is to create a system capable of verifying if 2 images of people have the face of the same person in them. Therefore the main objective can be broken down into two parts:

1. A Segmentation Algorithm: Given a single image, crop out the location of the face in it.
2. A Verification Algorithm: Given two already segmented faces, determine if they belong to the same person or not.

We limited our scope to only one face per image, and only photographic images.



# **Partial Report**

## **Data Used**
To achieve our goals, we will need a large and varied dataset with two important characteristics:

1. Annotation of the bounding box of the face present in the image.
2. Annotation of the identity of the person present in the image.

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset has both of those characteristics, and is also extremely large, having 202,599 images. There are two versions of it, one with the raw image, the other with the face already mostly cropped. We will use the raw version, and initially intend to use it exclusively. All the images are photographs of celebrities, in a variety of conditions, with their faces visible.

However, there are other possibly useful datasets. We currently don't intend to use them, but we may later. They are:

1. [WIDER](http://shuoyang1213.me/WIDERFACE/):

    * More varied them CelebA.
    * It lacks identity annotation, so it would only be useful for the segmentation aspect of the project.
    * Many images have more them one face in them, so filtering would be required.

2. [LFW](http://vis-www.cs.umass.edu/lfw/):

    * Specialized for verification, having many images per person.
    * It lacks bounding box annotation, so it would only be useful for the verification aspect of the project

### **CelebA Examples**
<img src="sample_imgs/raw/000002.jpg" width="250" height="250"></img>

*Image 1.1 - Imaging containing mostly the head*

<img src="sample_imgs/raw/000093.jpg" width="250" height="250"></img>

*Image 1.2 - Imaging containing entire body*

<img src="sample_imgs/raw/000041.jpg" width="250" height="250"></img>

*Image 2.1 - First Image of Individual 1058*

<img src="sample_imgs/raw/000050.jpg" width="250" height="250"></img>

*Image 2.2 - Second Image of Individual 1058*

<img src="sample_imgs/segmented/true/000041.jpg" width="250" height="250"></img>

*Image 3.1 - Image 2.1 segmented with bounding box annotation*

<img src="sample_imgs/segmented/true/000050.jpg" width="250" height="250"></img>

*Image 3.2 - Image 2.2 segmented with bounding box annotation*

We added part of the data used to this GitHub repository, and the rest to a [Google Drive folder](https://drive.google.com/open?id=1hyYXqt3cPbxsjqjWoT4iSwf806Kri0ic).



## **Planned Steps**
To achieve our goals, we plan to take the following steps:

### **1. Dataset Filtering and Preprocessing**
We will reorganize and reformat the original dataset to better suit our needs and improve usability. Importantly, for the face verification algorithm, we will only use individuals with more than 5 images of them in the dataset.

### **2. Image Preprocessing**
We plan to apply image processing techniques to facilitate the segmentation process.

We will, firstly, resize the image to a more manageable and constant size for our algorithms. We are going to use the Bicubic Interpolation to preserve the details.

Also, we will test changes to the color domain of our images, to see whether that has a positive effect on the results. We will test the Grayscale, RGB, and possibly the HSL domains.

Lastly, we intend to use manual edge detection algorithms, such as the Canny Edge Detector, to see if they improve our results.

### **3. Face Segmentation**
After preprocessing, we will input the resulting image to a bounding box detection algorithm to segment the face from the rest of the image. We will compare traditional methods to a *CNN* based method that we will implement, both in terms of speed and accuracy. We will test Depthwise Separable Convolutions to reduce computational costs.

### **4. Face Feature Vector Generation**
With the cropped face, we will create a CNN model that generates a feature vector for the face. Such vectors should be similar for different images of the same person. To achieve this, we will use a pre-trained VGG16 network, trained on a different but similar problem and dataset, and fine-tune the vectors generated by a hidden layer of it using the Triplet Loss.

### **5. Face Verification**
With the feature vector extracted, a threshold applied to either the euclidian or cosine distance of 2 vectors should be enough to verify the individual.



## **Initial Code and Results**
As of the writing of this report, we have mostly focused on a codebase for later usage, and processing and organizing the dataset. The code is located on [src](./src/), and the processed data on [data](./data/) and on [Google Drive](https://drive.google.com/open?id=1hyYXqt3cPbxsjqjWoT4iSwf806Kri0ic).

However, we do have some initial results. We ran a basic CNN model, with no hyperparameter optimization, no data augmentation, and no image preprocessing. We obtained the following results during training. The orange line is the training metric, and the blue one the validation metric.

<img src="sample_code/basic_cnn/epoch_loss.svg" height="350"></img>

*Image 4.1 - Mean Squared Error of bounding box location throughout training*

<img src="sample_code/basic_cnn/epoch_mean_absolute_error.svg" height="350"></img>

*Image 4.2 - Mean Absolute Error of bounding box location throughout training*

The resulting model was used to crop 100 images for visualization. Those are stored in [sample_imgs/segmented/pred](./sample_imgs/segmented/pred/). We did the same process for comparison using the annotation data, and stored in [sample_imgs/segmented/true](./sample_imgs/segmented/true/). The resulting crops don't seem as precise as is expected from the metrics, so we will analize what may be occuring.

<img src="sample_imgs/segmented/pred/000044.jpg" width="250" height="250"></img>

*Image 5.1 - An instance where the model had a very good result*

<img src="sample_imgs/segmented/true/000044.jpg" width="250" height="250"></img>

*Image 5.2 - Ground Truth for Image 5.1*

<img src="sample_imgs/segmented/pred/000040.jpg" width="250" height="250"></img>

*Image 6.1 - An instance where the model had bad but not terrible results. Most cases are like this*

<img src="sample_imgs/segmented/true/000040.jpg" width="250" height="250"></img>

*Image 6.2 - Ground Truth for Image 6.1*

<img src="sample_imgs/segmented/pred/000094.jpg" width="250" height="250"></img>

*Image 7.1 - An instance where the model had terrible results*

<img src="sample_imgs/segmented/true/000094.jpg" width="250" height="250"></img>

*Image 7.2 - Ground Truth for Image 6.1*

We also developed a notebook with the baseline face detection algorithm, located at [sample_code/FaceDetectionCV](./sample_code/FaceDetectionCV.ipynb).

<img src="sample_imgs/cascade_sample.png" width="250" height="250"></img>

*Image 8 - Baseline Face Detection Sample*

Finally, we implemented the canny edges detector, with a sample notebook at [sample_code/CannyEdgeExample](./sample_code/CannyEdgeExample.ipynb)

<img src="sample_imgs/canny_sample.png" width="250" height="250"></img>

*Image 9 - Canny Edge Detection Sample*



# **Final Report**

## **Data Used**

We decided to use the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset exclusevely. It has the annotations needed, and is also extremely large, having 202,599 images. There are two versions of it, one with the raw image, the other with the face already mostly cropped. We used the raw version. All the images are photographs of celebrities, in a variety of conditions, with their faces visible.



## **Steps**

### **1. Dataset Filtering and Preprocessing**
We reorganized and reformated the original dataset to better suit our needs and improve usability. Importantly, for the face verification algorithm, we only used individuals with more than 5 images of them in the dataset. Also, for the bounding box annotation, we changed the way the data is represented. Originally, the data is given as the coordinates of the upper-left point of the bounding box, and the width and height of the bounding box. We changed it to the coordinates of the upper-left point and the bootom-right point of the bounding box. This resulted in better metrics.



### **2. Image Preprocessing**

#### **Data augmentation**

Our model wasn't generating good results when the face was on the bottom part of the image. That is because in most images of the dataset, the face is on the upper part. To solve that, we used a data augmentation technique, by translating the image randomly, without taking the face bounding box out of the image. After the translation, the image would contain an empty part (section that wasn't on the original image). To fix that we filled those parts with the closest edge color of the original image.

<img src="https://i.ibb.co/fnch6V6/2.png" width="210" height="300" /> <img src="https://i.ibb.co/CQzzPnM/000002.jpg" width="210" height="300" />

*Image 1 - Translation Augmentation Example*

More examples can be found [here]("./sample_code/DataAugmentationTranslation.ipynb").

#### **Resizing**
Our CNN models only take inputs of a specific size. So we need to resize the image to the correct size. This resizing needs to be done for every bacth of training, and when executing, so we decided to use the bilinear interpolation for its relative low computational cost.



### **3. Face Segmentation**
We used CNN's to predict the bounding box of the face. Our CNN's receive an unsegmented image as input, and return 4 real number between 0 and 1. Those numbers are the relavant positions, repectevely, of the left, upper, right, and lower edges of the bouding box.

#### **Depthwise Separable Convolutions**
We used 2 different versions of our CNN's. One used standard convolutions, and the other used depthwise separable convolutions. Depthwise separable convolutions are a different operation, similar to regular convolutions, but with the aim to reduce number of parameters and, especially, number of operations. They do so by breaking up the channels of the input image, them performing a single convolution on each channel. Them, they recombine the resulting filtered images, and apply *N* regular convolutions, of size 1x1. This is best represented by image 2.

<img src="./sample_imgs/depthwise_sep.png" width="600" height="400"></img>

*Image 2 - Depthwise Separable Convolution Representation. [Original Source of the Image](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69)*

#### *Intersection Over Union*
To evaluate our models, we decided top use the intersection-over-union metric. It works by dividing the area of the intersection of the two bounding boxes by the are of their union.

<img src="./sample_imgs/iou.png" width="300" height="234"></img>

*Image 3 - IoU Representation. [Original Source of the Image](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png)*

For each type of convolution, we tried several different hyperparameters configurations, resulting in 54 experiments each. Here are the training graphs for the best models of each one.

<img src="./sample_imgs/cnn_train.svg" height="350"></img>

*Image 4 - Regular CNN IoU over training. Green line is validation data, and pink is training*

<img src="./sample_imgs/separable_train.svg" height="350"></img>

*Image 5 - Depthwise Separable CNN IoU over training. Red line is validation data, and blue is training*

We then chose the best ones to improve with the data augmentation and image preprocessing described above.



### **4. Face Feature Vector Generation**


### **5. Face Verification**

## **Code and Results**

**Comparison with OpenCV solution**

To compare the results of our Face Detector Models with OpenCV's Face Detector we used three different metrics: the Running Time, Intersection Over Union and Mean Absolute Error.

These tests was done by running both algorithms for all images found on sample_imgs/raw.

|                         | Regular Model     | Model using Depth-Wise Separable Convolution | OpenCV            |
|-------------------------|-------------------|----------------------------------------------|-------------------|
| Running time (seconds)  | 12.455038         | 7.462625                                     | 13.410666         |
| Intersection over Union | 0.74141621        | 0.73826022                                   | 0.69251639        |
| Mean Absolute Error     | 0.11603838        | 0.11802663                                   | 0.13947525        |

* The Mean Absolute Error is the sum of the absolute differences of all points of the predicted bounding boxes and the original bounding boxes points divided by the amount of points (4x amount of images).


---

**Keywords:** Image Segmentation; Feature Learning; Deep Learning; Faces; Face Verification; Triplet Loss;
