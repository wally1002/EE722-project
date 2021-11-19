# Crowd-counting-EE626-course-project

This is an implementation of the paper [CSRnet](https://arxiv.org/abs/1802.10062) for crowd counting.\
We have used Google colab to run all python notebooks.

## Dataset

[ShanghaiTech Dataset](https://drive.google.com/drive/folders/1bKs3w-KfFgyweDwVGpAR_QzCEuz6jm2q?usp=sharing)\
We have already created and saved the ground truths using gaussian filters from the linked [repo](https://github.com/davideverona/deep-crowd-counting_crowdnet).

## Ground Truth Generation
The ShaghaiTech dataset provides us with head annotations, this sparse matrix was converted in a 2D density map by using a Gaussian Filter. Then we sum all the values in the density map to get the actual count.

Use the `CSRnet_create_dataset.ipynb` to generate the ground truth's. \
Note - We have already saved the ground truth in this final datasets. \
If you are planning to run it make sure that the `root` variable is changed to the proper address.

## Training
The model architecture is divide into two parts, front-end and back-end. The front-end consists of 13 pretrained layers of the vgg16 model. The fully connected layers of the vgg16 are not taken. The back-end comprises of dilated convolution layers. The dilation rate at which maximum accuracy was obtained was experimentally found out be 2 as suggested in the CSRNet paper.

Use the `CSRnet_train.ipynb` to train the model and save checkpoints. We have made a 90-10 split to choose the best trained parameters.\
Note that you have to make the splits separately for Part A and B datasets so make sure to change the `task` variable to properly save the checkpoints(state of the model).\
Also change the `root` variable to the address of your folder.

## Testing
Use the `CSRnet_test.ipynb` to test the model using the saved model states.\
The `saves` folder contains the link to the trained parameters.\
Note - Change the `root` variable to the address of your folder.

## Results

|       Dataset       | MAE           |  
| ------------------- | ------------- |
|ShanghaiTech part A  | 88.232        | 
|ShanghaiTech part B  | 16.684        |

ShanghaiTech part A :

<img src="https://github.com/Dibyakanti/Crowd-counting-EE626-course-project/blob/main/img/A_test.png"> \

ShanghaiTech part B :

<img src="https://github.com/Dibyakanti/Crowd-counting-EE626-course-project/blob/main/img/B_test.png">

Note - The pictures are normalized in the two above examples

## References

1. [CSRnet](https://arxiv.org/abs/1802.10062) for crowd counting.
2. [Crowdnet](https://github.com/davideverona/deep-crowd-counting_crowdnet)
