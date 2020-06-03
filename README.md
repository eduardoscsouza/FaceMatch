# **FaceMatch - An Image Face Verifier**
Final project for **SCC0251/SCC5830 - Image Processing - 1st Semester of 2020** at University of São Paulo.

## **Abstract**
This project aims to build a system capable of, given two images, determine if both have the face of the same person or different people in them, known as the face verification problem. It will use image segmentation to separate the faces from the rest of the images, and image classification to determine whether the cropped faces belong to the same person. The images used will be photographic images of people, where their faces are visible, mainly from the WIDER and CelebA datasets. Possible applications revolve around security, where it is needed to verify someone's identity by an image of them, such as unlocking a phone and many surveillance systems.

## **Detailed Description**
This project is aimed at the face verification problem. It is divided into two parts, segmentation and verification.

The segmentation aspect is the main strictly image processing aspect of the project, and consists of generating bounding boxes of people's faces in an image, and subsequently cropping them. Currently, we are aiming at only one bounding box per image.
The verification aspect consists of getting two images of faces and determining if the faces belong to the same person.

For both aspects of the project, we intend to compare different approaches, both classical and deep learning ones, and analyze both accuracies as well as computational costs.

In terms of the deep learning models, for the segmentation we will attempt to use CNN's, with Depth-Wise Separable Convolutions for faster execution.
For the verification, we will attempt to fine-tune a pre-trained network, trained on a regular classification dataset, using the triplet loss. The idea is to get an extractor of highly separable feature vectors, and use those to distinguish the faces.

## **Example Images**
![](example_imgs/michael_1.jpg)

*Image 1 - First Image of Michael Jackson*

![](example_imgs/michael_2.jpg)

*Image 2 - Second Image of Michael Jackson*

![](example_imgs/taylor.jpg)

*Image 3 - Image of Taylor Swift*

The system built by this project should, in all images, be able to crop the faces out of them, and then identify the first and the second as belonging to the same person, and the third to a different person.

## **Authors**
 - Eduardo Santos Carlos de Souza (NUSP 9293481)
 - Guilherme Hideo Tubone (NUSP 9019403)

**Keywords:** Image Segmentation; Feature Learning; Deep Learning; Faces; Face Verification; Triplet Loss;