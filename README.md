
# Synthetic Promoter Analysis

We generated hundreds of thousands of random DNA sequences to drive gene expression, also known as promoters. These sequences ranged from ~100 base pairs to 500 base pairs. 
We placed a fluorescent reporter gene under each of these promoters, and added this promoter-fluorescent gene complex to human cells. 
Upon stimulation of the cell under certain conditions, we checked which of the promoters turned on its downstream fluorescent gene. 
We sorted the fluorescing cells into high and low expression populations, and sent them for DNA sequencing. 

This is a classic classification problem. How do we predict whether a sequence causes high-expression or low-expression? In order to explore this, we implement a variety of neural networks from the academic literature in the Keras framework. Once we have a model, we can interrogate our model for features like saliency, and whether the learned filters match with known DNA features. 
## Dataset
This dataset is very noisy. During the sorting of cells between the two populations there was significant mixing, resulting in improper labeling of the DNA sequences. This will likely hinder learning. In addition, cell cycle effects might also result in inaccurate labeling because although the cell  was labeled correctly *for that moment*, the label was not reflective of their typical state. 

## Comparison of Neural Network Implementations

We benchmark multiple different neural networks on our dataset, implemented directly from their academic papers. In addition, we frequently optimize the architecture to improve performance. Very broadly, they are the following: 
* [CNN-LSTM with an embedding layer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5747425/pdf/pone.0188129.pdf) to represent the information
* [CNN-LSTM with one-hot encoding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/)
* [Hilbert CNN](https://openreview.net/forum?id=HJvvRoe0W) where the sequential information is converted to a 3d 'image' using a Hilbert curve
* 1D [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
* Hilbert [MobileNet](https://arxiv.org/pdf/1704.04861.pdf), similar to MobileNet but uses a Hilbert space-filling curve for samples

#### CNN-LSTM with Embedding Layer
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/embedded_fig.png)
Here we tried to optimize the number of convolutions prior to the bidirectional LSTM, as well as the size of the MaxPool before the LSTM layers. 

![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/embedded_roc.png)

#### CNN-LSTM with One-Hot Encoding
Similar to the CNN-LSTM with an embedding layer, except we use one-hot encoding, which should reduce model complexity. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/cnnlstm_fig.png)
For this model we attempted to optimize the stride and kernel size of the MaxPooling layer, as well as the number of convolutions prior to the bidirectional LSTM. We also test a separable depthwise convolution. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/cnnlstm_roc.png)

#### 1D DenseNet 
Similar to the original DenseNet, only using 1D convolutions and max pooling. The major feature of DenseNet is the residual connections between every block. In addition, our version is not as deep, to prevent overfitting. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/densenet_fig.PNG)
We tuned the type of convolution and length of the stem block. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/1ddense_roc.png)

#### Hilbert CNN
Finally, we tested a Hilbert CNN. The Hilbert curve is a space fitting curve that we use to transform a 1D tensor to a 2D tensor. It creates a perfect square, so the order of a hilbert curve reflects the dimensions of the square it creates. E.g. Order of 3 = 2<sup>3</sup>  x 2<sup>3</sup> = 16 x 16 square.

 It is worth noting that the original authors did 4-mer one-hot encoding, meaning every set of 4 unique nucleotides got converted to a vector 256 long. My testing did not discern any difference between 4-mer and 1-mer one-hot encoding, so I optimized this using 1-mer encoded vectors which results in 4 channels, as opposed to 256 channels per sample, which can dramatically increase storage requirements and computation time. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/hilbert_curvetrunc.png)
Yin et al. benchmarked a variety of space fitting curves on different datasets, and Hilbert always came out ahead. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/mappingstrats.PNG)
The general architecture looks like this:
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/hilbertcnn.PNG)
With each computation block appears as follows:
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/hilbertcnn_residualblock.PNG)
We played around with a variety of parameters, such as activations at the end of computation blocks, use of max pooling prior to the fully connected layers, and dropout. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/hilbert_roc.png)
#### Truncated Hilbert MobileNet
Very similar idea to the Hilbert CNN, just using a different architecture. In addition, we truncate most of the upper layers in order to reduce model complexity. Architecture from the original paper: 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/mobilenet_fig.png)
See ROC curves from above to see this model compared with the Hilbert-CNN. 

## Determination of Feature Co-occurance

Sequences were analyzed using distance metrics against known features in the JASPAR database. 
We then determined which features co-occur with each other within a given sequence more frequently than expected.
This is similar to determining whether two words occur in the same sentence more frequently than expected, given a corpus. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/feature_cooc.png)
## TO DO
* Create ensemble model
* Generate saliency examples
* Show frequency of filters matching known important DNA sequences

## Authors

* **Wolfgang Rahfeldt** - *Initial work* - [FloatFlow](https://github.com/FloatFlow)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* All papers linked above
* Gabriel Altay for a Hilbert Curve Script - [Galtay](https://github.com/galtay)
