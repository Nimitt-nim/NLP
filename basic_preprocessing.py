import nltk
from nltk import word_tokenize, TweetTokenizer, MWETokenizer

text = "I ate 8.5 ice-creams in New Delhi 🥶😇"


#Problem with domain specific text#
tokens = word_tokenize(text)
print(tokens)

#Processing Tweets
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(text))

tokenizer = TweetTokenizer()
more_complex_text = "Congrats @shashitharoor Sir, for winning ‘Chevalier de la Legion d’Honneur’, highest civilian honour by the French government.#FranceinIndiafrancediplo_EN"
print(tokenizer.tokenize(more_complex_text))
print(word_tokenize(more_complex_text))



tokenizer = MWETokenizer()
tokenizer.add_mwe(('New', 'Delhi'))
tokenizer.tokenize(word_tokenize(text))


# Cleaning Text by removing non-alphabetical tokens

print("123".isalpha())
print("NLP".isalpha())


text = "I'm eating food and drinking milk."
tokens = word_tokenize(text)
print(tokens)
cleaned_tokens = [token for token in tokens if token.isalpha()]
print(cleaned_tokens)


# lowercasing
lowercased_tokens = [token.lower() for token in cleaned_tokens]
print(lowercased_tokens)

#Stopword Removal
from nltk.corpus import stopwords
print(stopwords.words("english"))
# print(stopwords.words("hindi"))
print(stopwords.words("russian"))
print(stopwords.fileids())
print(stopwords.words("hinglish"))


#Correcting Spelling Mistakes
from nltk.corpus import words
from nltk.metrics.distance import edit_distance
correct_words = words.words("en") # list of correctly spelled words
#print(correct_words)


misspelled_word = "instituon" # "oblied", "instituon" "scohol"
threshold = len(misspelled_word)
correctly_spelled_word = ''
for word in correct_words:
	if word[0] == misspelled_word[0]:
		editDis = edit_distance(word,misspelled_word)
		if  editDis < threshold:
			correctly_spelled_word = word
			threshold = editDis
print(correctly_spelled_word)


# Lemmatizer 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("dogs"))
print(lemmatizer.lemmatize("churches"))
print(lemmatizer.lemmatize("introduction"))

# Stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("dogs"))
print(stemmer.stem("introduction"))



