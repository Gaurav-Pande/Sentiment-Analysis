# Sentiment-Analysis

This Project now have 2 components:

1. Learn Sentiment analysis on Yelp reviews using pytorch deep learning models. The idea is to learn the basics of NLP.
2. A small project to compare Rule based and ML based sentiment analysis techniques(a binary classification problem)


### Contents:
- Yelp reviews sentiment analysis using Deep learning methods
  - Main Parts in Text analysis
  - How to Run
  - Analysis
- Rule based and ML based Methods
  - Introduction
  - Dataset Information
  - Requirements
  - Results
  - Rule Based Approach
    - VADER
    - SentiWordNet
  - ML based Approach
    - Naive Bayes Approach
  
# Yelp reviews sentiment analysis using Deep learning methods

The purpose of this notebook is to go through all the basics for an NLP task. The breakdown of the tasks would be as follows:


## Main Parts in Text analysis:

1. **Data Processing**: Process the raw data, convert it to a pandas dataframe, and manipulate the data according to your need and save it to another csv file.
2. **Data Vectorization**: The process of converting the text reviews to a vector of intergers using one hot encoding. Deep learning models do not accept any textual inputs rather you need to feed the inputs as intgers or floats.
3. **Data vocabulary**: we need to create a vocabulary for an NLP task, because our model can learn only from the words it has seen so far and their position in the text as well. so for this purpose we need to know which word come how many times in a text, where it appear in the text. we store all such information in a python dictionary
4. **Data processing in pytorch**: We process the data in pytorch in using torch dataloader by input our dataset, batch_size. It automatically converts the dataset in batches of tensors for us. so we need not to split the dataset in batches separately. It also handles the autograd for us. In short we are missing out:
  * Batching the data
  * Shuffling the data
  * Load the data in parallel using multiprocessing workers.
  
Dataloader provides all these functions.

5. **Deep Learning Model**: so far the models that I a working on are:
  * Single Layer Perceptron with following params:
      * One Linear Layer of Softmax
      * Sigmoid activation unit
      * Adam optimizer to upgrade the weights of the parameters
      * Binary cross entropy loss which deals with models which spits binary outputs.
  * Multi Layer Perceptron:
      * 2 Linear Layer
      * 1 sigmoid activation unit
      * Adam optimizer
      * Binary cross entropy loss
       

6. Training, Validation and Testing Loop.
7. Hyperparameters Tuning and their understanding.


## How to run:
To run the notebook, you can open the directory and run: 
```
cd yelp_reviews
jupyter notebook

```

You can also run the notebook in google collab and run the cell by cell. I have added the discriptions for each relevant models as we go through the notebook. Here is the link for the collab: [link](https://colab.research.google.com/drive/1yXJrOIF2EX-n6S2xZW5bLtxOosD-G2X1?usp=sharing)


## Analysis:

One of the thing that I have observed so far is:

* If I use the very light dataset, then the simple perceptron works really well.
* If I use the full dataset of yelp reviews the perceptron overfits and I get accuracy of 100 percent, and I can clearly observe that it is overfitting because when I print top 20 positive words, it spits random garbage.


[TODO]:
* Seed understanding as currently the output is changing [DONE]
* Use cuda in pytorch and re run the model : [DONE]
* Add code to store the vectorized data and model files.
* prediction of the model [DONE]
* other deep learning model implementation.
