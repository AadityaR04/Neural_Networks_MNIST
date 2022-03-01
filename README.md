# Neural Networks

## What are Neural Networks?

* Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms.

<img src="https://miro.medium.com/max/850/0*Ql4g_GiiEYZ7moZt.png" width="600">

[Image Source](https://aditya22pande.medium.com/industry-use-cases-of-neural-networks-d3a90ae69637)

* Neural networks are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

<img src="https://miro.medium.com/max/1302/1*UA30b0mJUPYoPvN8yJr2iQ.jpeg" width="600">

[Image Source](https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf)

<img src="https://miro.medium.com/max/828/1*t7z6V85E9mxPLpubcJTAGQ.jpeg" width="600">

[Image Source](https://medium.com/@zoey.yuzhu/cost-function-vs-loss-function-1546f4299365)

* The main aim in Neural Networks is to calculate the Loss and Cost functions in the Forward Propagation (or Forward Pass) and reduce the Cost by performing a Backpropagation to change the weights and biases

<img src="https://www.researchgate.net/publication/303744090/figure/fig3/AS:368958596239360@1464977992159/Feedforward-Backpropagation-Neural-Network-architecture.png" width="400">

[Image Source](https://www.researchgate.net/figure/Feedforward-Backpropagation-Neural-Network-architecture_fig3_303744090)

---

## What are Convolutional Neural Networks?

<img src="https://miro.medium.com/max/1400/1*uAeANQIOQPqWZnnuH-VEyw.jpeg" width="600">

[Image Source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

* In the last decade, deep learning has evolved into an extensive field within research and development. In image analysis, convolutional neural networks (CNN) have been particularly successful. 
* Convolutional Neural Networks refers to a subset of neural networks with a specific network architecture, where each so-called hidden layer typically has two distinct layers: the first stage is the result of a local convolution of the previous layer (the kernel has trainable weights), the second stage is a max-pooling stage, where the number of units is significantly reduced by keeping only the maximum response of several units of the first stage. After several hidden layers, the final layer is typically a fully connected layer. It has a unit for each class that the network predicts, and each of those units receives input from all units of the previous layer.

---

## What are MNIST Datasets?

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png">

[Image Source](https://en.wikipedia.org/wiki/MNIST_database)

* The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.
* The MNIST database contains 60,000 training images and 10,000 testing images.

----

## MNIST Digit Classifier

* We shall be using Neural Networks (Linear and Convolutional) and train them on the MNIST training dataset, and then utilize the trained network in identifying numbers/digit classification from the MNIST testing dataset

### Final Output

<img src="https://i.imgur.com/CbpaLB9.png" width="200">

## Tools used:

* [Python](https://www.python.org/)
* [Jupyter Notebook](https://jupyter.org/)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pytorch](https://pytorch.org/)
