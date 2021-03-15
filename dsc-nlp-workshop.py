#Imports:
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from nltk.probability import FreqDist
from collections import defaultdict

from heapq import nlargest


def getNonStopwords(myWords):
    #Return all words minus stop words and punctiation
    stopWords = set(stopwords.words('english') + list(punctuation))
    nonStopWords = []

    for myWord in myWords:
        if myWord not in stopWords:
            nonStopWords.append(myWord)

    return nonStopWords
    

def getSentenceRankings(sentences, freq):
    #For each sentence (i) go through all words in the sentence and look up how often it
    #occurs in the text overall (freq) this number is added to the sentence (i)'s sentence importance score

    sentenceScores = defaultdict(int)

    #for each sentence
    for i,sentence in enumerate(sentences):
        #for each word in the sentence
        for word in word_tokenize(sentence.lower()):
            if (word in freq):
                sentenceScores[i] += freq[word]

    return sentenceScores


def summarizeText(originalText, n):
        #1 Split text by sentences/break by period and place into an array
        sentences = sent_tokenize(originalText)
        if len(sentences) < n:
            return "Error: Article too small to summarize"

        #2 Strip text from punctuation and break into an array of individual words
        words = word_tokenize(originalText.lower())

        #3 Create a list of words found in the text that do not include punctuation or stop words (punctuation = . , ' " / etc. | stop words = and, or, the, is, are etc. )
        nonStopwords = getNonStopwords(words)

        #4 Create a frequency distribution of each unique word (how many times is each unique word present?) {sentence index : frequency}
        freq = FreqDist(nonStopwords)

        #5 Calculate using the frequency of each unique word, a score that tells us how many popular words each sentence has, {sentence index : score}
        sentenceScores = getSentenceRankings(sentences, freq)

        #6 Stores the top n {sentence index : score} entries based off 'sentenceScoores.get' (the score)
        nMostImportantSentencesIndexes = nlargest(n, sentenceScores, key=sentenceScores.get)

        #7 Sort the most popular sentences (the ones with the highest score) by their index so that they are back in a logical order
        sortedMostImportantSentencesIndexes = sorted(nMostImportantSentencesIndexes)

        #8 Piece together our most important sentences and return our summarized string!
        summarizedSentences = ""
        for index in sortedMostImportantSentencesIndexes:
            summarizedSentences += sentences[index] + " "

        return summarizedSentences
        


if __name__ == "__main__":
    #originalText = input("Enter a text to summarize: ")
    
    with open('inputText.txt', 'r') as file:
        originalText = file.read().replace('\n', '')
    print(originalText)

    n = input("enter n: ")
    summarizedText = summarizeText(originalText, int(n))
    print('\n\n -------Summarized Text:------- \n\n' + summarizedText)
    