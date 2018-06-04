from nltk.sentiment import vader
import codecs
import unicodedata
import pandas as pd

file_path_pos = "/Users/gpande2/nltk_data/cornell_movie_review_data/rt-polaritydata/rt-polaritydata/rt-polarity-pos.txt"
file_path_neg = "/Users/gpande2/nltk_data/cornell_movie_review_data/rt-polaritydata/rt-polaritydata/rt-polarity-neg.txt"

#lines = codecs.open(file_path_pos, 'r', encoding='utf-8').readlines()

with open(file_path_pos, 'rb') as f:
    postive_lines = [l.decode('utf8', 'ignore') for l in f.readlines()]

with open(file_path_neg, 'rb') as f:
    negative_lines = [l.decode('utf8', 'ignore') for l in f.readlines()]

ob =vader.SentimentIntensityAnalyzer()
def vaderSentiment(review):
    return ob.polarity_scores(review)['compound']


def getReviewSentiment(vaderSentiment):
    positive_sentiments_scores = [vaderSentiment(review=review) for review in postive_lines]
    negative_setiment_socres = [vaderSentiment(review=review) for review in negative_lines]
    return {'positive-review-score-list':positive_sentiments_scores,'negative-review-score-list':negative_setiment_socres}



def calAccuracy(review_results):
    positive_review_result = pd.Series(review_results['positive-review-score-list'])
    negative_review_result =pd.Series(review_results['negative-review-score-list'])
    df = pd.DataFrame({'positive':positive_review_result,'negative':negative_review_result})
    df_filtered_positive = df['positive'][df['positive']>0]
    df_filtered_negative = df['negative'][df['negative']<0]
    print ('Accuracy of the positive review sentiment analysis is %.2f' %((len(df_filtered_positive)/len(positive_review_result))*100))
    print ('Accuracy of the negative review sentiment analysis is %.2f' %((len(df_filtered_negative)/len(negative_review_result))*100))
    print ('Total accuracy of the analysis is %.2f' %(((len(df_filtered_positive) + len(df_filtered_negative))/(len(positive_review_result)+len(negative_review_result)))*100))


def runDiagnostics(reviewResult):
    positiveReviewsResult = reviewResult['positive-review-score-list']
    negativeReviewsResult = reviewResult['negative-review-score-list']
    pctTruePositive = float(sum(x > 0 for x in positiveReviewsResult))/len(positiveReviewsResult)
    pctTrueNegative = float(sum(x < 0 for x in negativeReviewsResult))/len(negativeReviewsResult)
    totalAccurate = float(sum(x > 0 for x in positiveReviewsResult)) + float(sum(x < 0 for x in negativeReviewsResult))
    total = len(positiveReviewsResult) + len(negativeReviewsResult)
    print ("Accuracy on positive reviews = " +"%.2f" % (pctTruePositive*100) + "%")
    print ("Accurance on negative reviews = " +"%.2f" % (pctTrueNegative*100) + "%")
    print ("Overall accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")



# calAccuracy(getReviewSentiment())
# runDiagnostics(getReviewSentiment())

# a = [-1,-2,-3]
# print (sum(x<0 for x in a),sum(a))
# print([x<0 for x in a])

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from string import punctuation
# a = swn.senti_synsets('dog')
# print (list(swn.senti_synsets('dog')))

def superNaiveSentimentAnalysis(review):
    reviewpolarity = 0.0
    numExceptions = 0.0
    for word in review.lower().split():
        weight = 0.0
        try:
            common_meaning = list(swn.senti_synsets(word))[0]
            if common_meaning.pos_score()>common_meaning.neg_score():
                weight = weight + common_meaning.pos_score()
            elif common_meaning.pos_score()<common_meaning.neg_score():
                weight = weight - common_meaning.neg_score()
        except:
            numExceptions = numExceptions + 1
        reviewpolarity = reviewpolarity + weight
    return reviewpolarity

runDiagnostics(getReviewSentiment(superNaiveSentimentAnalysis))

stopwords=set(stopwords.words('english')+list(punctuation))

def NaiveSentimentAnalysis(review):
 reviewPolarity = 0.0
 numExceptions = 0
 for word in review.lower().split():
   numMeanings = 0
   if word in stopwords:
     continue
   weight = 0.0
   try:
     for meaning in swn.senti_synsets(word):
       if meaning.pos_score() > meaning.neg_score():
          weight = weight + (meaning.pos_score() - meaning.neg_score())
          numMeanings = numMeanings + 1
       elif meaning.pos_score() < meaning.neg_score():
          weight = weight - (meaning.neg_score() - meaning.pos_score())
          numMeanings = numMeanings + 1
   except:
       numExceptions = numExceptions + 1
   if numMeanings > 0:
     reviewPolarity = reviewPolarity + (weight/numMeanings)
 return reviewPolarity

runDiagnostics(getReviewSentiment(NaiveSentimentAnalysis))
