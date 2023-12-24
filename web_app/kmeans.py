import io
import json
import requests
import os, shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA

path = r"C:\Users\Linggar Maretva\Desktop\New folder"
df_movies = pd.read_csv(r'%s\static\files\movie_synopsis.csv' % path)
index_drop = df_movies[df_movies['synopsis'] == "No overview found."].index
df_movies.drop(index_drop , inplace=True)
df_movies.reset_index(drop =True, inplace=True)
vectorizer = pickle.load(open(r'%s\static\files\vectorizer.sav'  % path, 'rb'))
features = pickle.load(open(r'%s\static\files\features.sav'  % path, 'rb'))

def train(k) :
	vectorizer = pickle.load(open(r'%s\static\files\vectorizer.sav'  % path, 'rb'))
	features = pickle.load(open(r'%s\static\files\features.sav'  % path, 'rb'))
	kmeans_model = KMeans(n_clusters = k)
	synopsis_clusters = kmeans_model.fit(features)
	# Save Model
	pickle.dump(kmeans_model, open(r'%s\static\files\kmeans_model.sav' % path, 'wb'))
	
	# Save Labeled Data
	df_movies['label'] = kmeans_model.labels_
	with io.open(r'%s\static\files\movie_synopsis_labeled.csv' % path, 'w', encoding='utf-8') as f:
		df_movies.to_csv(f)
	
	# Save Data for Each Cluster
	folder = (r'%s\static\files\csv cluster' % path)
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print(r'Failed to delete %s. Reason: %s' % (file_path, e))
	
	clusters = df_movies.groupby('label')
	for cluster in clusters.groups :
		f = open((r'%s\static\files\csv cluster\cluster' % path)+str(cluster)+ '.csv', 'w', encoding='utf-8') # buat file csv untuk tiap cluster
		data = clusters.get_group(cluster)[['title','synopsis']] # judul dan sinposis tiap data pada tiap cluster
		f.write(data.to_csv(index_label='id')) # simpan ke csv
		f.close()
	
	# Save Feature Names
	order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1] # diurutkan berdasarkan indeks -> lalu di-reversed
	terms = vectorizer.get_feature_names()
	n_terms = 10

	feature_names = [[0 for x in range(n_terms)] for y in range(k)] 
	for i in range(k) :
		j_counter = 0
		for j in order_centroids [i, :n_terms] :
			print('   %s' % terms[j])
			feature_names[i][j_counter] = terms[j]
			j_counter += 1
	with io.open(r'%s\static\files\feature_names.csv' % path, 'w') as f:
		pd.DataFrame(feature_names).to_csv(f)
	
	# Save WordCloud
	c = []
	keywords = []
	for i in range(k) :
		c.append(i)
		key = ""
		for j in order_centroids [i, :n_terms] :
			key = key+(" ")+(terms[j])
		keywords.append(key)

	for i in range(k) :
	  text = keywords[i]
	  wordcloud = WordCloud(max_font_size=50, background_color="white").generate(text)
	  plt.figure()
	  plt.imshow(wordcloud, interpolation="bilinear")
	  plt.axis("off")
	  plt.savefig(r'%s\static\files\wordcloud\wordcloud%d.jpg' % (path, i))
	
	# Save Silhouette
	with open(r'%s\static\files\silhouette.txt' % path, 'w') as f:
		f.write(str(silhouette_score(features, labels = kmeans_model.labels_)))
	
	# # Save PCA
	# pca = PCA(n_components = 2)
	# reduced_features = pca.fit_transform(features.toarray())
	# reduced_cluster_centers = pca.transform(synopsis_clusters.cluster_centers_)
	
	# plt.figure(figsize=(30, 15), dpi=80)
	# plt.scatter(reduced_features[:, 0], reduced_features[:,1], c = kmeans_model.predict(features))
	# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
	# plt.savefig(r'%s\static\files\2d.png' % (path))
	
	# pca_3d = PCA(n_components = 3)
	# reduced_features_3d = pca_3d.fit_transform(features.toarray())
	# reduced_cluster_centers_3d = pca_3d.transform(synopsis_clusters.cluster_centers_)
	
	# sns.set(style = "darkgrid")
	# fig = plt.figure(figsize=(30, 15), dpi=80)
	# ax = fig.add_subplot(111, projection = '3d')
	# x = reduced_features_3d[:, 0]
	# y = reduced_features_3d[:, 1]
	# z = reduced_features_3d[:, 2]
	# ax.scatter(x, y, z, c = synopsis_clusters.labels_)
	# ax.scatter(reduced_cluster_centers_3d[:, 0], reduced_cluster_centers_3d[:,1], reduced_cluster_centers_3d[:,2], marker='x', s=500, c='b')
	# plt.savefig(r'%s\static\files\3d.png' % (path))
	
	

def silhouette() :
	silhouette = open(r'%s\static\files\silhouette.txt' % path, 'r').read()
	return silhouette

def cluster_number() :
	filename = r'%s\static\files\feature_names.csv' % path
	feature_names = pd.read_csv(filename, header=0)
	feature_names = list(feature_names.values)
	index_counter = 0
	index = []
	for i in feature_names :
	  index.append(i[0])
	  index_counter += 1
	return index

def labeled_data() :
	filename = r'%s\static\files\movie_synopsis_labeled.csv' % path
	movie_synopsis_labeled = pd.read_csv(filename, header=0)
	movie_synopsis_labeled = movie_synopsis_labeled.head()
	movie_synopsis_labeled = list(movie_synopsis_labeled.values)
	return movie_synopsis_labeled

def feature_names() :
	filename = r'%s\static\files\feature_names.csv' % path
	feature_names = pd.read_csv(filename, header=0)
	feature_names = list(feature_names.values)
	return feature_names

def clusters_data() :
	filename = r'%s\static\files\feature_names.csv' % path
	feature_names = pd.read_csv(filename, header=0)
	feature_names = list(feature_names.values)
	index_counter = 0
	index = []
	for i in feature_names :
	  index.append(i[0])
	  index_counter += 1
	cluster_number = len(index)
	  
	clusters_data_full = []
	for i in range(cluster_number) :
		path_file = r"%s\static\files\csv cluster" % path
		filename = path_file + "\cluster" + str(i) + ".csv"
		clusters_data = pd.read_csv(filename, header=0)
		clusters_data = clusters_data.head()
		clusters_data = list(clusters_data.values)
		clusters_data_full.append(clusters_data)
	return clusters_data_full

def predict_cluster(sentence) :
	vectorizer = pickle.load(open(r'%s\static\files\vectorizer.sav'  % path, 'rb'))
	kmeans_model = pickle.load(open(r'%s\static\files\kmeans_model.sav' % path, 'rb'))
	Y = vectorizer.transform([sentence])
	prediction = kmeans_model.predict(Y)
	cluster_prediction = prediction[0]
	
	query = [sentence.lower()]
	query = vectorizer.transform(query)
	
	# Get Cosine Similarity Score
	features = pickle.load(open(r'%s\static\files\features.sav'  % path, 'rb'))
	cosine_score = []
	index = 0
	for i in features :
	  cosine_score.append(cosine_similarity(query, i))
	  index +=1

	# Get Sorted Index Cosine Similarity
	cosine_score_update = []
	for i in cosine_score :
		cosine_score_update.append(i[0][0])
	cosine_score_update = np.array(cosine_score_update)
	cosine_score_update
	indices = np.argsort(cosine_score_update)[::-1]

	related_movie = []
	for i in range(10) :
		index_now = indices[i]
		if (cosine_score_update[index_now] != 0) :
			movie_array = [df_movies['title'][index_now], cosine_score_update[index_now], df_movies['synopsis'][index_now]]
			related_movie.append(movie_array)
	
	return cluster_prediction, related_movie

def predict_cluster_bytitle(query) :

	# Get Title & Synopsis from Query
	title_new = ""
	synopsis_ori = ""
	url = (r"https://api.themoviedb.org/3/search/movie?api_key=63fea4c709da1f1496b7a1ca7a3c6083&language=en-US&query=%s&page=1&include_adult=false" % query)
	r = requests.get(url)
	json_data = json.loads(r.text)
	try:
		if (json_data['results'][0]['overview'] != "") :
		  title_new = json_data['results'][0]['title']
		  synopsis_ori = json_data['results'][0]['overview']
	except Exception:
		pass

	# Predict Cluster
	vectorizer = pickle.load(open(r'%s\static\files\vectorizer.sav'  % path, 'rb'))
	kmeans_model = pickle.load(open(r'%s\static\files\kmeans_model.sav' % path, 'rb'))
	Y = vectorizer.transform([synopsis_ori])
	prediction = kmeans_model.predict(Y)
	cluster_prediction = prediction[0]
	
	query = [synopsis_ori.lower()]
	query = vectorizer.transform(query)
	
	# Get Cosine Similarity Score
	features = pickle.load(open(r'%s\static\files\features.sav'  % path, 'rb'))
	cosine_score = []
	index = 0
	for i in features :
	  cosine_score.append(cosine_similarity(query, i))
	  index +=1

	# Get Sorted Index Cosine Similarity
	cosine_score_update = []
	for i in cosine_score :
		cosine_score_update.append(i[0][0])
	cosine_score_update = np.array(cosine_score_update)
	cosine_score_update
	indices = np.argsort(cosine_score_update)[::-1]

	related_movie = []
	for i in range(10) :
		index_now = indices[i]
		if (cosine_score_update[index_now] != 0) :
			movie_array = [df_movies['title'][index_now], cosine_score_update[index_now], df_movies['synopsis'][index_now]]
			related_movie.append(movie_array)

	return title_new, synopsis_ori, cluster_prediction, related_movie