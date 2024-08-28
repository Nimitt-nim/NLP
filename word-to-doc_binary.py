from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.corpus import words
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys




print(reuters.fileids(),len(reuters.fileids()))
print(reuters.categories(),len(reuters.categories()))

# #for fileid in reuters.fileids():
# #	print(fileid,reuters.raw(fileid),"...")
# #	exit()


def cosine_sim(vec1,vec2):
	if len(vec1) == len(vec2):
		score = round(dot(vec1,vec2)/(norm(vec1)*norm(vec2)),2)
		return score
	else:
		return "Dimensions do not match!"


print("entering processing")
stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Following lines of code will create dictionary: Key: articleID, Value: set of clean tokens in the article
count = 0 
processed_corpus= {}
for fileid in reuters.fileids():
	text = reuters.raw(fileid)
	text = text.lower()
	tokens = word_tokenize(text)
	cleaned_tokens = [token for token in tokens if token.isalpha()]
	processed_corpus[count] = set(cleaned_tokens) - stopwords
	processed_corpus[count] = set(map(lemmatizer.lemmatize, processed_corpus[count]))
	count = count + 1

#for key in processed_corpus:
#	print(key,processed_corpus[key][:10])

print("entering word2doc preparation")

# Following lines of code will create dictionary: Key: word, Value: set of articleID where this word is present
word_to_doc_sparse_dict = {}
for key in processed_corpus:
	for word in processed_corpus[key]:
		if word not in word_to_doc_sparse_dict:
			word_to_doc_sparse_dict[word] = set()
		word_to_doc_sparse_dict[word].add(key)	
			

print("entering printing word2doc")
for word in word_to_doc_sparse_dict:
	print(word,word_to_doc_sparse_dict[word])


print(word_to_doc_sparse_dict["car"])

print("word_to_doc_vectorization start")
#converting sparse representation

# Following lines of code will create dictionary: Key: word, Value: a list of length "no. of articles" 
vectorized_word_to_doc = {}
for word in word_to_doc_sparse_dict:
	vectorized_word_to_doc[word] = [0]*len(processed_corpus)
	for doc_id in word_to_doc_sparse_dict[word]:
		vectorized_word_to_doc[word][doc_id] = 1

print(cosine_sim(vectorized_word_to_doc["car"],vectorized_word_to_doc["truck"]))
print(cosine_sim(vectorized_word_to_doc["ship"],vectorized_word_to_doc["captain"]))
print(cosine_sim(vectorized_word_to_doc["car"],vectorized_word_to_doc["captain"]))
print(cosine_sim(vectorized_word_to_doc["potato"],vectorized_word_to_doc["vegetable"]))
print(cosine_sim(vectorized_word_to_doc["potato"],vectorized_word_to_doc["ship"]))
print(cosine_sim(vectorized_word_to_doc["ash"],vectorized_word_to_doc["chemical"]))

print(round(sys.getsizeof(vectorized_word_to_doc["car"])/1024,2),"KB")
