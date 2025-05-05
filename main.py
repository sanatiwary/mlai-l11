# content based recommendation: based on genre
# natural language preprocessing:
# 1) remove stop words - ex. Remi is a good girl traveling to Dubai -> Remi good girl travel Dubai
# 2) count vectorizer - turning words to numbers and comparing similarities
# 3) Tf IDP vectorizer - measure of originality of a word by comparing the num of times word (d) appears in a document with num of docs the word appears in, Tf (term frequency) IDF (inverse document frequency) = Tf(d) * IDF(d)
# 4) cosine similarities -> finds score using dot product

import pandas as pd

movieData = pd.read_csv("movies_metadata.csv")
print(movieData.head())

from sklearn.feature_extraction.text import TfidfVectorizer
tfIDF = TfidfVectorizer(stop_words="english")

movieData["overview"] = movieData["overview"].fillna("")
tfIDFMatrix = tfIDF.fit_transform(movieData["overview"])

print(tfIDFMatrix.shape)

print(tfIDF.get_feature_names_out()[5000:5010])

from sklearn.metrics.pairwise import linear_kernel
cosineSim = linear_kernel(tfIDFMatrix, tfIDFMatrix)

indices = pd.Series(movieData.index, index=movieData["title"]).drop_duplicates()

def getRec(title):
    idx = indices[title]

    simScores = list(enumerate(cosineSim[idx]))
    simScores = sorted(simScores, key=lambda x:x[1], reverse=True)
    simScores = simScores[1:11]

    movieIndices = [i[0] for i in simScores]

    return movieData[title].iloc[movieIndices]

print(getRec("The Dark Knight Rises"))
