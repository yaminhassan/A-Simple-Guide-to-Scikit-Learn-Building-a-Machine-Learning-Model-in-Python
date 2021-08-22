import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as sk
from sklearn.feature_extraction.text import TfidfVectorizer as Tf
df = pd.read_csv("IMDB Dataset.csv")
df_positive = df[df["sentiment"] == "positive"][:9000]
df_negative = df[df["sentiment"] == "negative"][:1000]
df_imb = pd.concat([df_positive, df_negative])
rus = RandomUnderSampler(random_state=0)
df_bal, df_bal["sentiment"] = rus.fit_resample(df_imb[["review"]], df_imb["sentiment"])
print(df_bal.head())
print(df_bal.sentiment.head())


train, test = sk(df_bal, test_size=0.33, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']
tfidf = Tf(stop_words="english")
train_x_vector = tfidf.fit_transform(train_x)
x = pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index, columns=tfidf.get_feature_names())
test_x_vector = tfidf.transform(test_x)
svc = SVC(kernel="linear")
svc.fit(train_x_vector, train_y)
print(svc.predict(tfidf.transform(['A good movie'])))
print(svc.predict(tfidf.transform(['An excellent movie'])))
print(svc.predict(tfidf.transform(['I did not like this movie at all'])))

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)
print(dec_tree.predict(tfidf.transform(['nice'])))
print(dec_tree.predict(tfidf.transform(['An excellent movie'])))
print(dec_tree.predict(tfidf.transform(['I did not like this movie at all'])))
print(train_x)
