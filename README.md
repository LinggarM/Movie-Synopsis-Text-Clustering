# Movie-Synopsis-Text-Clustering
Movie Synopsis Text Clustering using K-Means Clustering and TF-IDF Vectorizer and deployment using framework Flask

## About the Project

This project is a major assignment project for the second semester of the natural language processing course. The objective of this task is to perform text clustering on movie synopsis data and transform it into a web application.

## Technology Used

  * Python
  * Pandas
  * Numpy
  * Matplotlib
  * Seaborn
  * Scikit-learn
  * Wordcloud
  * Requests

## Notebook File
* [synopsis_clustering.ipynb](notebooks/synopsis_clustering.ipynb)

## Workflow

### Data Collection
  - The dataset used in this project is the movie synopsis data obtained using **[The Movie Database (TMDB) API](https://developer.themoviedb.org/reference/intro/getting-started)**. The retrieved data includes movie titles and their synopses only, while genres are not included because they will be predicted in an unsupervised manner by the k-means model during the text clustering process.
  - Despite not including the genres feature, the movie data collected consists of films from various diverse genres. There are a total of **19 genres** across all the movie data collected, and the list of these genres is as follows:
    - Drama
    - Crime
    - Comedy
    - Action
    - Thriller
    - Documentary
    - Adventure
    - Science Fiction
    - Animation
    - Family
    - Romance
    - Mystery
    - Horror
    - Fantasy
    - War
    - Music
    - History
    - Western
    - TV Movie 
  - The quantity of collected data is **8214** movie and synopsis data
  - The dataset is stored in [data/movie_synopsis.csv](data/movie_synopsis.csv)

### Data Preprocessing
  The data preprocessing steps applied to the data include:
  - **Remove missing values**: Delete instances with missing values, such as movies lacking any synopsis text.
  - **Case folding**: Transform all letters into lowercase.
  - Train the **TF-IDF Vectorizer** model.

### Model Training
  - The model is trained using the **K-Means** algorithm to perform clustering on movie synopsis data
  - The training process is conducted using a parameter **k (number of clusters)** set to **14**. Although the original number of genres for movies is 19, this value serves as an initial parameter, which will later be evaluated using the **Elbow Method** and **Silhouette Score**.
  - The trained model is saved to [models/kmeans_model.sav](models/kmeans_model.sav)
  - After the training model is completed, each data point is assigned a label based on its cluster number, and then saved to the file [data/movie_synopsis_labeled.csv](data/movie_synopsis_labeled.csv).
  - **Data distribution based on cluster:**

    ![images/data_distribution.png](images/data_distribution.png)

    Note: There's imbalanced data on cluster 6
  - Each cluster has its own feature names, which are words located at the centroid of each cluster, representing the genre of that cluster. Ten feature names are selected for each cluster, and then saved to file [data/feature_names.csv](data/feature_names.csv).
  - **The following are word cloud representations of feature names for each cluster.**
  
    Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3
    :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
    ![wordcloud0](images/wordcloud/wordcloud0.jpg)  |  ![wordcloud1](images/wordcloud/wordcloud1.jpg)  |  ![wordcloud2](images/wordcloud/wordcloud2.jpg)  |  ![wordcloud2](images/wordcloud/wordcloud3.jpg)
    Cluster 4 | Cluster 5 | Cluster 6 | Cluster 7
    ![wordcloud4](images/wordcloud/wordcloud4.jpg)  |  ![wordcloud5](images/wordcloud/wordcloud5.jpg)  |  ![wordcloud6](images/wordcloud/wordcloud6.jpg)  |  ![wordcloud6](images/wordcloud/wordcloud7.jpg)
    Cluster 8 | Cluster 9 | Cluster 10 | Cluster 11
    ![wordcloud8](images/wordcloud/wordcloud8.jpg)  |  ![wordcloud9](images/wordcloud/wordcloud9.jpg)  |  ![wordcloud10](images/wordcloud/wordcloud10.jpg)  |  ![wordcloud10](images/wordcloud/wordcloud11.jpg)
    Cluster 12 | Cluster 13
    ![wordcloud12](images/wordcloud/wordcloud12.jpg)  |  ![wordcloud13](images/wordcloud/wordcloud13.jpg)
  - For the next steps, the trained model can be used to:
    - **Predict clusters** based on the entered movie synopsis.
    - Provide **movie recommendations based on the entered synopsis** (by calculating cosine similarity between the entered synopsis and the list of movies in the database, and returning movies that are close to the entered synopsis).
    - Provide **movie recommendations based on the entered title** (by obtaining the synopsis from the entered movie title, then performing the same steps as in the previous point, i.e., calculating cosine similarity).
### Model Evaluation
  - **Elbow Method** (SSE):

    ![images/elbow_method.png](images/elbow_method.png)
  
  - **Silhouette Score**:
    - 0.004184973145629372
    - **Note**: Silhouette score ranges from -1 to 1, where a higher score indicates better-defined clusters.

### Model Visualization
  - **2D PCA (Principal Component Analysis (PCA))**:

    - ![images/2d_pca.png](images/2d_pca.png)
  
  - **3D PCA (Principal Component Analysis (PCA))**:
  
    - ![images/3d_pca.png](images/3d_pca.png)

## Installation

Change `path` variable in `kmeans.py` into where you going to save this repo. For the example I am using `C:\Users\Linggar Maretva\Desktop\Movie-Synopsis-Text-Clustering` as my path.
```sh
...
...
from sklearn.decomposition import PCA

path = r"C:\Users\Linggar Maretva\Desktop\Movie-Synopsis-Text-Clustering"
df_movies = pd.read_csv(r'%s\static\files\movie_synopsis.csv' % path)
...
...
```

## Usage

Run by typing these code :
```sh
python app.py
```

## Screenshots
<img src="https://github.com/LinggarM/Movie-Synopsis-Text-Clustering/raw/main/Movie%20Synopsis%20Clustering%20Web%20App/static/assets/img/ss.PNG"/>

## Publication
* [Aplikasi Clustering Film Berdasarkan Sinopsisnya.pdf](docs/Laporan_PBA_Kelompok%205.pdf)

## Contributors
* [Linggar Maretva Cendani](https://github.com/LinggarM) - [linggarmc@gmail.com](mailto:linggarmc@gmail.com)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
