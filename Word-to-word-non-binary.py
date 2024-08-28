from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.corpus import reuters
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import words
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import sys


def cosine_sim(w1,w2):
	stemmed_word1 = stemmer.stem(w1)
	stemmed_word2 = stemmer.stem(w2)
	if stemmed_word1 in vectorized_word_to_word:
		if stemmed_word2 in vectorized_word_to_word:
			vec1 = vectorized_word_to_word[stemmed_word1]
			vec2 = vectorized_word_to_word[stemmed_word2]
		else:
			return w2 + " is not present in corpus"
	else:
		return w1 + " is not present in corpus"
	if len(vec1) == len(vec2):
		score = round(dot(vec1,vec2)/(norm(vec1)*norm(vec2)),2)
		return score


#print(reuters.fileids(),len(reuters.fileids()))
#print(reuters.categories(),len(reuters.categories()))

#for fileid in reuters.fileids():
#	print(fileid,reuters.raw(fileid)[:200],"...")

print("entering processing")
stopwords = set(stopwords.words("english"))
#lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Following lines of code will create dictionary: Key: sentenceID, Value: set of clean tokens in the sentence

count = 0 
processed_corpus= {}
for fileid in reuters.fileids():
	text = reuters.raw(fileid)
	text = text.lower()
	sentences = sent_tokenize(text)
	for sentence in sentences:
		tokens = word_tokenize(sentence)
		cleaned_tokens = [token for token in tokens if token.isalpha()]
		processed_corpus[count] = set(cleaned_tokens) - stopwords
		processed_corpus[count] = set(map(stemmer.stem, processed_corpus[count]))
		count = count + 1

#for key in processed_corpus:
#	print(key,processed_corpus[key][:10])

print("entering word2word preparation")

# Following lines of code will create dictionary: Key: word, Value: list of context words with whom this word is present
word_to_word_sparse_dict = {}
for key in processed_corpus:
	for word in processed_corpus[key]:
		if word not in word_to_word_sparse_dict:
			word_to_word_sparse_dict[word] = list()
		word_to_word_sparse_dict[word].extend(processed_corpus[key])	
	
word_to_int = {}
count = 0		
for word in word_to_word_sparse_dict:
	word_to_int[word] = count
	count = count + 1 
#print("entering printing word2doc")
#for word in word_to_word_sparse_dict:
#	print(word,word_to_word_sparse_dict[word])


#print(word_to_word_sparse_dict["car"])

print("word_to_word_vectorization start")
#converting sparse representation

# Following lines of code will create dictionary: Key: word, Value: a list of length "no. of words in the corpus" 
vectorized_word_to_word = {}
for word in word_to_word_sparse_dict:
	vectorized_word_to_word[word] = [0]*len(word_to_int)
	for context_word in word_to_word_sparse_dict[word]:
		vectorized_word_to_word[word][word_to_int[context_word]] += 1

print(cosine_sim("car","truck"))
print(cosine_sim("ship","captain"))
print(cosine_sim("car","captain"))
print(cosine_sim("ash","chemical"))
print(cosine_sim("food","potato"))
print(cosine_sim("hospital","doctor"))
print(cosine_sim("hospital","patient"))
print(cosine_sim("university","student"))
print(cosine_sim("university","professor"))
print(cosine_sim("student","professor"))
print(cosine_sim("student","doctor"))




print(round(sys.getsizeof(vectorized_word_to_word["car"])/1024,2),"KB")
