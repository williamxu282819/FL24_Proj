How can crowd wisdom be effectively realized, especially when the crowd is overconfident? Representing such crowds with analogous ensembles of many convolutional neural networks (CNNs), I explore different methods of aggregating their prediction outcomes to improve their confidence calibration. I propose a novel crowd aggregation approach: re-learning a combined set of high-dimensional neural representations with independent, low-dimensional decision models, such as random forests. Then, I validate the approach on MNIST and CIFAR-10 datasets across various ensemble sizes, training samples and positive-evidence (PE) conditions simulated through different contrast and noise manipulations. The experiment results demonstrate superior confidence calibration across all scenarios, indicating that the crowd wisdom of neural networks may be more latent than explicit.

Reading the **crowds.csv** in crowd_summary_stat/different_images or /same_images.

network: 1-15, identifies the 15 networks in a given crowd;

crowd: 1-100, identifies each of the 100 crowds

image: this merely an indicator for the 1-300 images used for each network, in numerical order

0_cnn ~ 9_cnn: a probability score (0-1) that the neural network thinks the true label is a particular label from 0 to 9 (10 options)

predicted_label_cnn: the neural network's prediction; this is the label class with the highest probability score provided in 0_cnn to 9_cnn.

cnn_model_confidence: a probability score (0-1) that the neural network believes that its prediction is correct; please note that this model_confidence assumes an underlying distribution with softmax algorithm, and it's different from confidence_cnn in that it's a two-choice question: is the network predict correct? yes/no.

confidence_cnn: the highest probability score provided among 0_cnn to 9_cnn.



0_rf ~ 9_rf: a probability score (0-1) that the random forest, based on the neural network's latent representations, thinks the true label is a particular label from 0 to 9 (10 options). A neural network provides 300 latent representations for each of the 300 images, and 225 such representations are used for training the random forest, while 75 such representations (with 75 corresponding images) are used for testing the random forests. 

For the 225 representations that are used for training the random forest, no probability scores are calculated for the 225 corresponding images, and are left with NANs.

predicted_label_rf: the random forest's prediction; this is the label class with the highest probability score provided in 0_rf to 9_rf.

confidence_rf: the highest probability score provided among 0_rf to 9_rf.


MNIST_index: this is a unique image identifier. I can use this MNIST_index to trace down an actual image from the MNIST dataset from which the images are sampled.