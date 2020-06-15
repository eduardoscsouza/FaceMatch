# **FaceMatch - An Image Face Verifier**
Final project for **SCC0251/SCC5830 - Image Processing - 1st Semester of 2020** at University of São Paulo.



## **Authors**
 - Eduardo Santos Carlos de Souza (NUSP 9293481)
 - Guilherme Hideo Tubone (NUSP 9019403)



## **Abstract**
This project aims to build a system capable of, given two images, determine if both have the face of the same person or different people in them, known as the face verification problem. It will use image segmentation to separate the faces from the rest of the images, and image classification to determine whether the cropped faces belong to the same person. The images used will be photographic images of people, where their faces are visible, mainly from the CelebA dataset. Possible applications revolve around security, where it is needed to verify someone's identity by an image of them, such as unlocking a phone and many surveillance systems.



## **Main Objective**
As stated in the abstract, our objective is to create a system capable of verifying if the 2 images of people have the face of the same person in them or not. Therefore the main objective can be broken down into two parts:

1. A Segmentation Algorithm: Given a single image, crop out the location of the face in it. We will limit our scope to only one face per image.
2. A Verification Algorithm: Given two already segmented faces, determine if they belong to the same person or not.



## **Data Used**
To achieve or goals, we will need large and varied datasets with two important characteristics:

1. Annotation of bounding boxes of the face present in the image. As stated before, we will work with only a single face per image.
2. Annotation of the identity of the person present in the image.

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset has both those characteristics, and is also extremely large, having 202,599 images. There are two versions of it, one with the raw image, the other with the face already mostly cropped. We will use the raw version, and initially intend to use it exclusevely.

However there are other possibly useful datasets. We curently don't intend to use them, but we may later. They are:

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

We added part of data used to this GitHub repository, and the rest to a [Google Drive folder](https://drive.google.com/open?id=1hyYXqt3cPbxsjqjWoT4iSwf806Kri0ic).



## **Planned Steps**
To achieve our goals, we plan to take the following steps:

### **1. Dataset Filtering and Preprocessing**
We will reorganize and reformat the original dataset to better suit our needs and improve usability. Importantly, for the face verification, we will only use individuals with more than 5 images of them in the dataset.

### **2. Image Preprocessing**
We plan to apply image processing techniques to facilitate the segmentation process.
We will, firstly, resize the image to a more manageble and constant size for our algorithms. We are going to use the Bicubic interpolation to preserve many of the details.
Also, we will test changes to the color domain of our images, to see wether that has a positive effect on the results. We will test the Grayscale, RGB and possibly the HSL domains.
Lastly, we intend to use manual edge detection algorithms, such as the Canny Edge Detector, to see if they improve our results.

### **3. Face Segmentation**
After preprocessing, we will input the resulting image to a bounding box detection algorithm to segment the face from the rest of image. We will compare traditional methods to a *CNN* based method that we will implement, both in terms of speed and accuracy. We will test Depth-Wise Separable Convolutions to reduce computational costs.

### **4. Face Feature Vector Generation**
With the cropped face, we will create a CNN model that is able to generate a feature vector for the face. Such vectors should be similar for different images of the same person.
To achieve this, we will use a pre-trained VGG16 network, trained on a different but similar problem and dataset, and fine tune the vectors generate by a hidden layer of it using the Triplet Loss.

### **5. Face Verification**
With the feature vector extracted, a threshold applied to either the euclidian or cossine distance of 2 vectors should be enough to verify the individual.



## **Initial Code and Results**
As of the writting of this report, we have mostly focous on a code base for later use, and processing and organizing the dataset. The code is located on [src](./src/), and the processed data on [data](./data/) and on [Google Drive](https://drive.google.com/open?id=1hyYXqt3cPbxsjqjWoT4iSwf806Kri0ic).

However we do have some initial results. We ran a basic CNN model, with no hyper parameter optimization and no data augmentation or enhancement. We obtained the following results during training. The orange line is the training metrics, and the blue one the validation metric.

<img src="sample_code/basic_cnn/epoch_loss.svg" width="700" height="250"></img>

*Image 4.1 - Mean Squared Error of bounding box location through training*

<img src="sample_code/basic_cnn/epoch_mean_absolute_error.svg" width="700" height="250"></img>

*Image 4.2 - Mean Absolute Error of bounding box location through training*

The resulting model was used to crop 100 images for visualization. Those are stored in [sample_imgs/segmented/pred](./sample_imgs/segmented/pred/), and we use did the same using the annotation data, and stored in [sample_imgs/segmented/true](./sample_imgs/segmented/true/). The resulting crops don't seem as precise as is expected from the metrics, so we will analize what may be occuring.



---

**Keywords:** Image Segmentation; Feature Learning; Deep Learning; Faces; Face Verification; Triplet Loss;