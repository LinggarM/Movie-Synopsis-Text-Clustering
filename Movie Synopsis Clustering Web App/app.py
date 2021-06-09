from flask import Flask, request, render_template, send_from_directory, abort, redirect, url_for
import kmeans
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		# Retrieve Data
		cluster = request.form['cluster']
		kmeans.train(int(cluster))
		# Render
		# return render_template('index.html')
		return redirect(url_for('clusteringresult'))
	return render_template("index.html", movie_synopsis_labeled = kmeans.labeled_data(), silhouette = kmeans.silhouette())

@app.route('/clustering-result', methods=['GET', 'POST'])
def clusteringresult():
	return render_template("clustering_result.html", movie_synopsis_labeled = kmeans.labeled_data(), silhouette = kmeans.silhouette())

@app.route('/feature-names', methods=['GET', 'POST'])
def featurenames():
	return render_template("feature_names.html", feature_names = kmeans.feature_names())

@app.route('/data-per-clusters', methods=['GET', 'POST'])
def dataperclusters():
	return render_template("data_per_clusters.html", clusters_data = kmeans.clusters_data(), cluster_number = kmeans.cluster_number())

@app.route('/cluster-prediction', methods=['GET', 'POST'])
def clusterprediction():
	if request.method == 'POST':
		synopsis = request.form['synopsis']
		return render_template('cluster_prediction.html', synopsis = synopsis, cluster = kmeans.predict_cluster(synopsis)[0], related_movie = kmeans.predict_cluster(synopsis)[1])
	return render_template("cluster_prediction.html")

@app.route('/cluster-prediction-bytitle', methods=['GET', 'POST'])
def clusterpredictionbytitle():
	if request.method == 'POST':
		query = request.form['query']
		title, synopsis, cluster, related_movie = kmeans.predict_cluster_bytitle(query)
		return render_template('cluster_prediction_bytitle.html', query = query, title = title, synopsis = synopsis, cluster = cluster, related_movie = related_movie)
	return render_template("cluster_prediction_bytitle.html")

if __name__ == "__main__":
    app.run()