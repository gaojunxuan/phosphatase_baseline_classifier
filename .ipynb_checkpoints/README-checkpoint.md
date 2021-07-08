# Phosphatase Baseline Classifier
A baseline classifier for prediction of phosphatase activity.

In this project, we constructed and evaluated different non-neural models for classification. This established a set of baseline models for prediction of phosphatase activity. The codes are expected to be reusable for other datasets as well.

## Acknowledgement

The code for GP model is adapted from channels (https://github.com/fhalab/channels) and GPModel (https://github.com/yangkky/gpmodel). The repository *channels* contains code to reproduce the paper *Machine learning-guided channelrhodopsin engineering enables minimally-invasive optogenetics*, in which Gaussian process models are used for optimizing channelrhodopsin properties. 


## File structure

The repository is divided into two self-contained directories containing all the code and inputs for the regression and classification models, respectively. For regression, the GP code is here. The gpmodel module used in classification is cloned from https://github.com/yangkky/gpmodel. The regression code for phosphatase activity is from https://github.com/anaqiafendi/channels.