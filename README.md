# Covid-recognition
The COVID-19 pandemic which started in late 2019 has caused the world lots of distress and loss of lives. COVID-19 is a respiratory-related disease that mainly attacks the lungs and causes breathing difficulties amongst other symptoms. In this project, we aim to apply deep learning to detect the disease using a curated dataset of chest X-ray images. There are 3 classes in this dataset: Normal, COVID-19, and Viral Pneumonia. Hence, this is a multi-class classification problem. 

## Data
A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. In the current release, there are 1200 COVID-19 positive images, 1341 normal images, and 1345 viral pneumonia images. All images are grayscale and in png format.

[Link to the data](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

## Modeling
We used deep convolutional neural networks for modeling. With our _base model_, we achieved the following results on the test set:
Accuracy: 0.98  |  Precision: 0.98  |  Recall: 0.98  |  F1-Score: 0.98

We used transfer learning with _Inception-resnet V2_ and _Xception_ from the Keras library and our best model was the Xception model with the following results:
Accuracy: 0.99  |  Precision: 0.99  |  Recall: 0.99  |  F1-Score: 0.99

## Model Interpretability

We also explored model interpretability. It is vital to establish some trust in such a model especially as this problem statement is a health use case. We achieved this by exploring 2 options. The first is the [Grad-CAM](https://arxiv.org/abs/1610.02391) algorithm that outputs a heatmap (red being the highest intensity and blue, the lowest intensity) of the localized regions that are modt activated by the neural network in making the appropriate predictions. The second approach explored was analyzing and visualizing intermediate output feature maps from the first layer, down to the final layer.

Some Grad-CAM outputs are shown below:
![image](https://user-images.githubusercontent.com/37366301/117698396-50d98280-b189-11eb-90ad-dc2a8efe61e0.png)


For a Pneumonia test image, and for a select number of layers and select activations per layer, the intermediate output feature maps are shown below:
![image](https://user-images.githubusercontent.com/37366301/117698476-66e74300-b189-11eb-94b8-7e113e07a026.png)


## Deployment
We used Flask alongside web design tools like HTML and CSS to deploy our model as a light-weight web application. A future scope is to deploy the model in a production environment so it can be well-integrated into a hospital's technical architecture for a Radiologist's use.

Please feel free to suggest improvements to this project to any of the contributors!
