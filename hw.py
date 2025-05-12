import pandas as pd

retailData = pd.read_csv("Online Retail.csv")
retailData["Description"] = retailData["Description"].fillna("")

from sklearn.feature_extraction.text import TfidfVectorizer
tfIDF = TfidfVectorizer(stop_words="english")

tfIDFMatrix = tfIDF.fit_transform(retailData["Description"])

from sklearn.metrics.pairwise import linear_kernel
cosineSim = linear_kernel(tfIDFMatrix, tfIDFMatrix)

indices = pd.Series(retailData.index, index=retailData["Description"]).drop_duplicates()

def getRec(description):
    idx = indices[description]

    simScores = list(enumerate(cosineSim[idx]))
    simScores = sorted(simScores, key=lambda x:x[1], reverse=True)
    simScores = simScores[1:11]

    itemIndices = [i[0] for i in simScores]

    return retailData["Description"].iloc[itemIndices]

print(getRec("WHITE HANGING HEART T-LIGHT HOLDER"))
