from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

CORPUS = [] ## Put email messages here
y = []      ## Tag emails as spam/not-spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(CORPUS)

classifier = MultinomialNB() 
classifier.fit(X, y)

print(classifier.predict([])) ## Use the machine learning model here!
