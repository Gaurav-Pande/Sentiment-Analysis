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



# Rule based and ML based Methods

## Introduction:

This project is built to study how sentiment analysis works and does a comparision on the approaches followed to build a python based sentiment analysis application. 

Sentiment Analysis refers to the use of text analysis and natural language processing to identify and extract subjective information in textual contents. There are two type of user-generated content available on the web – facts and opinions. Facts are statements about topics and in the current scenario, easily collectible from the Internet using search engines that index documents based on topic keywords. Opinions are user specific statement exhibiting positive or negative sentiments about a certain topic. Generally opinions are hard to categorize using keywords. Various text analysis and machine learning techniques are used to mine opinions from a document [1]. Sentiment Analysis finds its application in a variety of domains.

Business Businesses may use sentiment analysis on blogs, review websites etc. to judge the market response of a product. This information may also be used for intelligent placement of advertisements. For example, if product “A” and “B” are competitors and an online merchant business “M” sells both, then “M” may advertise for “A” if the user displays positive sentiments towards “A”, its brand or related products, or “B” if they display negative sentiments towards “A”. Government Governments and politicians can actively monitor public sentiments as a response to their current policies, speeches made during campaigns etc. This will help them make create better public awareness regarding policies and even drive campaigns intelligently. Financial Markets Public opinion regarding companies can be used to predict performance of their stocks in the financial markets. If people have a positive opinion about a product that a company A has launched, then the share prices of A are likely to go higher and vice versa. Public opinion can be used as an additional feature in existing models that try to predict market performances based on historical data.

## Dataset Information
 
I have used the dataset build by the cornell university movie reviews dataset. This dataset contains 10,000 reviews on the movie. I have attached the dataset in the rt-polaritydata directory folder inside this repository. This data set contains 5000 positive reviews and 5000 negative reviews. I have built a rule based binary classifier(using corpus VADER and SentiwordNet) and ml based classifier(Naive Bayes classifier) to do the comparision between the two.
 
You can also download the datasets from the cornell wesite as well: [Link](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
 
 
## Requirements

The only library required for this project is python nltk library.


## Results:

I build the rule based classifier and used VADER and sentiwornet corpus and than determine their accuracy, and than i followed the ml based Naive bayes classifier to train and test the dataset. 

![Image](https://github.com/Gaurav-Pande/Sentiment-Analysis/blob/master/assets/RESULT.png?raw=true)

You can see that clearly the approach followed by ML is more accurate and robust in comparision to the rule based classifier.
The reason for this increase in accuracy for the ML based classifier is because the ml based classifer are built while training the model using "context we want to train". To understand this, lets take an example: we know that whale can belong to the category of mammal(Members of the infraorder Cetacea) or it can belong to the category of fish as well. So if a feature is whale feeded to the ml classifier what should it declare it? Should it declare it a mammal or fish? The interesting thing here is the we can train the ml model here with the example or contexts to figure out a new feature sets. For example we can train the model saying: if it moves like a fish and if it looks like a fish declare it a fish. so we can build our model in the context we want them to be build.


## Naive bayes algorithm
Take another example: You are walking in a park and suddenly a person crosses in front of you in a flash. Now you have given 2 option to tell who was that person. Was he a runner? or was he a cop?
how would you determine. Well if you know that it was a city marathon, than you would say that he was a runner or if you would have seen the appearances(say he is having a gun, a badge and a handcuffs) than you would immediately say he was a cop despite the fact the day in the city was a marathon. You see you train you mind with the context you would want to train and that is the same principle that has been used in the ml Naive Bayes algorithm, which uses the bayes theorem to determine the probabilty whether the person is a cop or runner given that the day was a city marathon or the appearances. using the probability we train the model to predict the outcomes on the test data sets. The name Naive can be confusing but it is not so naive as well :)


## Rule based approach:
In a rule based approach you use a set of rules or corpus to determine whether a review is positive or negative. This corpus is made manually by humans through a series of long study over the words and than programmers collect and bind them all into one known as sentiment lexicon(In terms of python it is basically a dictionary, but to lookup polarity). There are many sentiment lexicons available today like: 
* SentiWordNet
* VADER(It is actually both an algo and dataset)
* MPQA
* LIWC 

### VADER:

It is a rule base classifier built by georgia tech. Known as Valence Aware Dicionary for Sentiment Reasoning.It contains both algorithm and dataset. It is build on python. 
The algorithm takes text, calculates polarity and valence using rules. The dataset contains the lexicons.

### SentiwordNet

It is a sentiment lexicon with information of polarity. It is also built in nltk. Very convenient to use and it also extends Wordnet.

## ML Based Approach:

In this approach i divided the cornell movie review dataset into 2 parts. one from 0-2500 index(both positive and negative reviews) as a training data set and remaining is used to test the model and perform the accuracy on the model.

Remember in ml base approach we train the corpus with ml classifier and than test it to predict on new problem sets.

### Naive Bayes Algorithm:

It is based on the Bayes theorem and it that is why it comes under the family of "probabilistic classifiers".It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.

Bayes’ Theorem

Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

 P(A|B) = {P(B|A)* P(A)}/{P(B)} 

where A and B are events and P(B) ? 0.

Basically, we are trying to find probability of event A, given the event B is true. Event B is also termed as evidence.
P(A) is the priori of A (the prior probability, i.e. Probability of event before evidence is seen). The evidence is an attribute value of an unknown instance(here, it is event B).
P(A|B) is a posteriori probability of B, i.e. probability of event after evidence is seen.


In terms of movie review dataset we can say our bayes theorem will looks like this:


P(Positive/t) 	=    				P(t/Positive) x P(Positive)/
 					P(t/Positive) x P(Positive) + P(t/Negative) x P(Negative)


where t is any review or text.

similary for negative review:

P(Negative/t) =					P(t/Negative) x P(Negative)
					P(t/Positive) x P(Positive) + P(t/Negative) x P(Negative)
                    
The above is to find probability that the review is actually negative, given the text of the review (use Bayes Theorem)

we can see above that denominator of both the p(positive/t) and p(negative/t) is same, so we can compare only the numerators.


## References:

* pluralsite course: https://app.pluralsight.com/library/courses/building-sentiment-analysis-systems-python/table-of-contents
* Naive Bayes : https://www.geeksforgeeks.org/naive-bayes-classifiers/




[TODO]:
* Seed understanding as currently the output is changing [DONE]
* Use cuda in pytorch and re run the model : [DONE]
* Add code to store the vectorized data and model files.
* prediction of the model [DONE]
* other deep learning model implementation.
