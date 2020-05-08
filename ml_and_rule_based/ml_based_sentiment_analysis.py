import nltk
# Splitting the corpus into 2 parts one for training by ml classifier and another to test the model
file_path_pos = "/Users/gpande2/nltk_data/cornell_movie_review_data/rt-polaritydata/rt-polaritydata/rt-polarity-pos.txt"
file_path_neg = "/Users/gpande2/nltk_data/cornell_movie_review_data/rt-polaritydata/rt-polaritydata/rt-polarity-neg.txt"

splitindex = 2500

with open(file_path_pos, 'rb') as f:
    postive_lines = [l.decode('utf8', 'ignore') for l in f.readlines()]

with open(file_path_neg, 'rb') as f:
    negative_lines = [l.decode('utf8', 'ignore') for l in f.readlines()]


test_positive_reviews = postive_lines[splitindex+1:]
test_negative_reviews = negative_lines[splitindex+1:]

train_positive_reviews = postive_lines[:splitindex]
train_negative_reviews = negative_lines[:splitindex]


# create a set of vocabulary for all words
def getVocabulary():
    positiveWordList = [word for line in train_positive_reviews for word in line.split()]
    negaativeWordList = [word for line in train_negative_reviews for word in line.split()]
    allWordList  = [item for sublist in [positiveWordList,negaativeWordList] for item in sublist]
    vocabulary = list(set(allWordList))
    return vocabulary


# Use the vocabulary to extract features,i.e express each review in form of a tuple
# Basically you check if a word in a review exist in the vocabulary or not, if exit you make it true else false

def extract_features(review):
    review_words = set(review)
    features = {}
    vocabulary = getVocabulary()
    for word in vocabulary:
        features[word] = (word in review_words)
    return features

## Transforming Data to input to nltk
# it is a list of tuples first element  in each tuple is the review and the second element is the correcponding label
# to tell whether the word is positive or negatice
def getTrainingData():
    positiveTaggedData = [{'review':oneReview.split(),'label':'positive'} for oneReview in train_positive_reviews]
    negativeTaggedData = [{'review':oneReview.split(),'label':'negative'} for oneReview in train_negative_reviews]
    fullTaggedTrainingData = [item for sublist in [positiveTaggedData,negativeTaggedData] for item in sublist]
    trainingData = [(review['review'],review['label'])for review in fullTaggedTrainingData]
    return trainingData



def getTrainedNBClassifier(extract_features,trainingData):
    # the below will convert the reviews into the list of feature vectors
    traininfFeatures = nltk.classify.apply_features(extract_features,trainingData)
    # training the classifier
    trainingNBClassifier = nltk.NaiveBayesClassifier.train(traininfFeatures)
    return trainingNBClassifier

trainedNBClassifier = getTrainedNBClassifier(extract_features,getTrainingData())

# using the classifier to test the test data

def naiveBasedSentimentCalculater(review):
    problemInstance = review.split()
    problemFeature = extract_features(problemInstance)
    return trainedNBClassifier .classify(problemFeature)


def getTestReviewSentiments(naiveBasedSentimentCalculater):
    testNegResult = [naiveBasedSentimentCalculater(review) for review in test_negative_reviews]
    testPosResult = [naiveBasedSentimentCalculater(review) for review in test_positive_reviews]
    labelToNum = {'positive':1,'negative':-1}
    numericNegResult = [labelToNum[x] for x in testNegResult]
    numericPosResult = [labelToNum[x] for x in testPosResult]
    return {'result-on-positive':numericPosResult,'result-on-negative':numericNegResult}



def runDiagnostics(reviewResult):
    positiveReviewsResult = reviewResult['result-on-positive']
    negativeReviewsResult = reviewResult['result-on-negative']
    pctTruePositive = float(sum(x > 0 for x in positiveReviewsResult))/len(positiveReviewsResult)
    pctTrueNegative = float(sum(x < 0 for x in negativeReviewsResult))/len(negativeReviewsResult)
    totalAccurate = float(sum(x > 0 for x in positiveReviewsResult)) + float(sum(x < 0 for x in negativeReviewsResult))
    total = len(positiveReviewsResult) + len(negativeReviewsResult)
    print ("Accuracy on positive reviews = " +"%.2f" % (pctTruePositive*100) + "%")
    print ("Accurance on negative reviews = " +"%.2f" % (pctTrueNegative*100) + "%")
    print ("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")


if __name__ == '__main__':
    print ("executing Naive Bayes algorithm")
    runDiagnostics(getTestReviewSentiments(naiveBasedSentimentCalculater))
    print ("finished execution!!")