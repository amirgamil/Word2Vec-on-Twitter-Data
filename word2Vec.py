import gensim 
import nltk
from nltk import word_tokenize, sent_tokenize
import json
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#Loading & Processing Data
fname = get_tmpfile("vectors.kv")
def read_twitter_data():
	all_twitter_data = []
	with open("../../tweets.json", "r") as f: 
		for tweet in f:
			jsonTweet = json.loads(tweet)
			if jsonTweet["lang"] == "en":
				text = jsonTweet["text"]
				all_twitter_data.append(text.lower())
	return all_twitter_data

def process_data(data):
	input_data_for_model = []
	all_words = []
	for title in data:
		sentencesInData = list(sent_tokenize(title))
		wordsInData = word_tokenize(title)
		for word in list(wordsInData):
			all_words.append(word)
		for sentence in sentencesInData:
			input_data_for_model.append(list(word_tokenize(sentence)))
	return (all_words, input_data_for_model)


#Creating vocabulary of model
def create_model(input_data):
	model = gensim.models.Word2Vec(input_data, min_count=100, size=65, workers=3, sg =0, sample=1e-4)
	#Training model
	model.train(input_data, total_examples=len(input_data), epochs=10)
	word_vectors = model.wv
	model.save("vectorOfTweets.model")
	#Saving keyed vectors
	word_vectors.save(fname)


def load_model(fname):
	model = gensim.models.Word2Vec.load("vectorOfTweets.model")
	word_vectors = KeyedVectors.load(fname, mmap='r')
	return (model, word_vectors)

twitter_data = read_twitter_data()
word, input_data = process_data(twitter_data)
all_words = nltk.FreqDist(word)
print(all_words)
create_model(input_data)
model, word_vectors = load_model(fname)
#print(model.wv.most_similar("meningitis", topn=100))
keys=[]
embedding_vectors=[]

keys_similar_words = []
embedding_vectors_similar_words = []

for (word, _) in model.most_similar("meningitis", topn=30):
	keys_similar_words.append(word)
	embedding_vectors_similar_words.append(word_vectors[word])

for word in list(model.wv.vocab):
	keys.append(word)
	embedding_vectors.append(word_vectors[word])


tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
vectors_in_2d = tsne_model_en_2d.fit_transform(embedding_vectors)

vectors_in_2d_similar_words = tsne_model_en_2d.fit_transform(embedding_vectors_similar_words)

def tsne_plot_2d(label, embeddings, words=[], a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("2D Representation.png")

#tsne_plot_2d('Word Bank from Tweets Related to Meningitis', vectors_in_2d, keys, a=0.1)
tsne_plot_2d('30 Most Similar Words to Meningitis', vectors_in_2d_similar_words, keys_similar_words, a=0.1)

def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    fig.savefig("3D Representation.png")

tsne_model_en_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
embeddings_wp_3d = tsne_model_en_3d.fit_transform(embedding_vectors)

tsne_plot_3d('Visualizing Embeddings using t-SNE', 'Twitter data containing words related to meningitis', embeddings_wp_3d, a=0.1)

