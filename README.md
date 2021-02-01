# Unet-Segmentation-of-images
This project is basically Generation and Segmentation of synthetic data for surface Inspection on glass panels of Oven and prediction on real Images. In this code I mainly focused on the line, points and scratches that can be present on a surface. 

Dataset:
For training a convolutional neural network enough data is not available. So, I created a synthetic images randomly which contains lines and point defects along with their masks.
X_train & y_train are training images with their masks.
X_valid & y_valid are validation images with their masks.
The input image is in RGB and mask are of Gray scale. 

## Model:
I have used a U-Net architecture, it is a CNN architecture for fast and precise segmentation of images. it consists of contracting and expansive path. Input image size is of 128 * 128 * 3 and Output of size 128 * 128 * 1.

## Training:
 I have trained on 40 images and validate on 10 images i.e., split ratio of 0.2. I used binary cross entropy loss function and model is trained for 50 epochs..

## Prediction:
The real images were captured on the glass panels of oven using illuminations. Final stage involves the prediction real images using the trained model on synthetically created images. The results are shown below and looks satisfactory. 

Result:
**I have to increase the contrast on the predicted defects to visualize it clearly.

<img width="336" alt="Results" src="https://user-images.githubusercontent.com/77510506/106403435-067c9300-642f-11eb-89b3-9e08d83e3e25.PNG">


Reference:
https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
