# Synthetic Promoter Analysis

We generated hundreds of thousands of random DNA sequences to drive gene expression, also known as promoters. These sequences ranged from ~100 base pairs to 500 base pairs. 
We placed a fluorescent reporter gene under each of these promoters, and added this promoter-fluorescent gene complex to human cells. 
Upon stimulation of the cell, we checked which of the promoters turned on its downstream fluorescent gene. 
We sorted the fluorescing cells into high and low expression populations, and sent them for DNA sequencing. 

This naturally leads two a couple of questions. First, can we predict which DNA sequences will perform better in this context? 
Towards that end, we implement several prominant neural networks directly from the academic literature in the Keras framework.
Second, among the high performing DNA sequences, are there interactions between sequence features that suggest a mechanism?
In other words, for our corpus of DNA sequences, do certain features co-occur in a single sequence more frequently than expected?
For this, we perform more traditional analyses for feature identification and correlation.

Note that all datasets will not be included until publication. 

## Comparison of Neural Network Implementations

We benchmark multiple different neural networks on our dataset, implemented directly from their academic papers. 
* [CNN-LSTM with an embedding layer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5747425/pdf/pone.0188129.pdf) to represent the information
* [CNN-LSTM with one-hot encoding](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/)
* [CNN](https://openreview.net/forum?id=HJvvRoe0W) where the sequential information is converted to a 3d 'image' using a Hilbert curve
  * 1-mer: where every character is one-hot encoded
  * 4-mer: where every 4 characters are one-hot encoded, resulting in very sparse matrix

![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/nn_comparison.png)

Using a plain CNN with Hilbert preprocessing appears to give the best results, ~5% improvement over a CNN-LSTM using one-hot encoding. 
Since the 4-mer Hilbert preprocessing requires an enormous amount of data storage (every unique set of 4 characters is represented by a vector of length 256),
the 1-mer Hilbert method will be used in the future (every character is just a vector of length 4). Note that accuracy never exceeds ~75%. 
This is likely due to poor physical seperation of the two classes during biological experiments, resulting in mixing of the two classes. 
Future experiments will avoid this pitfall. 


## Determination of Feature Co-occurance

Sequences were analyzed using distance metrics against known features in the JASPAR database. 
We then determined which features co-occur with each other within a given sequence more frequently than expected.
This is similar to determining whether two words occur in the same sentence more frequently than expected, given a corpus. 
![alt text](https://github.com/FloatFlow/SyntheticPromoter/blob/master/readme_figures/feature_cooc.png)
## Authors

* **Wolfgang Rahfeldt** - *Initial work* - [FloatFlow](https://github.com/FloatFlow)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* All papers linked above
* Gabriel Altay for a Hilbert Curve Script - [Galtay](https://github.com/galtay)
