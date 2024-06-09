import os
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from .forms import TrainForm, PredictForm

# Paths to save the trained models and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_PATH = os.path.join(BASE_DIR, 'main/models/vectorizer.pkl')
KNN_MODEL_PATH = os.path.join(BASE_DIR, 'main/models/knn_model.pkl')
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, 'main/models/kmeans_model.pkl')

# Custom KNN Classifier
class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_neighbors]
            y_pred.append(max(set(nearest_labels), key=nearest_labels.count))
        return np.array(y_pred)

# Custom KMeans Clustering
class KMeansClustering:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(x)
            new_centroids = [np.mean(cluster, axis=0) if cluster else np.random.rand(X.shape[1]) for cluster in clusters]
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        return [np.argmin([np.linalg.norm(x - c) for c in self.centroids]) for x in X]

@csrf_exempt
def index(request):
    train_form = TrainForm()
    predict_form = PredictForm()
    return render(request, 'main/predictor.html', {'train_form': train_form, 'predict_form': predict_form})

@csrf_exempt
def train(request):
    if request.method == 'POST':
        train_form = TrainForm(request.POST, request.FILES)
        if train_form.is_valid():
            data_file = request.FILES['data_file']
            try:
                data = pd.read_csv(data_file)
                if 'text' not in data.columns or 'spam' not in data.columns:
                    return JsonResponse({'error': 'CSV file must contain "email_text" and "label" columns.'}, status=400)
                
                vectorizer = TfidfVectorizer(stop_words='english')
                X = vectorizer.fit_transform(data['text']).toarray()
                y = data['spam'].values

                # Train KNN
                knn = KNNClassifier(k=5)
                knn.fit(X, y)

                # Train KMeans
                kmeans = KMeansClustering(n_clusters=2)
                kmeans.fit(X)

                # Ensure the models directory exists
                os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)

                # Save models and vectorizer
                with open(VECTOR_PATH, 'wb') as f:
                    pickle.dump(vectorizer, f)
                with open(KNN_MODEL_PATH, 'wb') as f:
                    pickle.dump(knn, f)
                with open(KMEANS_MODEL_PATH, 'wb') as f:
                    pickle.dump(kmeans, f)

                return JsonResponse({'status': 'Training successful'})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
    else:
        train_form = TrainForm()
    return redirect('index')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        predict_form = PredictForm(request.POST)
        if predict_form.is_valid():
            email_text = predict_form.cleaned_data['email_text']
            k_value = predict_form.cleaned_data['k_value']

            # Check if model files exist
            if not os.path.exists(VECTOR_PATH) or not os.path.exists(KNN_MODEL_PATH) or not os.path.exists(KMEANS_MODEL_PATH):
                return JsonResponse({'error': 'Model files not found. Please train the model first.'}, status=500)

            with open(VECTOR_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(KNN_MODEL_PATH, 'rb') as f:
                knn = pickle.load(f)
            with open(KMEANS_MODEL_PATH, 'rb') as f:
                kmeans = pickle.load(f)

            X_new = vectorizer.transform([email_text]).toarray()

            knn_pred = knn.predict(X_new)[0]
            kmeans_pred = kmeans.predict(X_new)[0]

            return JsonResponse({
                'knn_prediction': 'spam' if knn_pred == 1 else 'not spam',
                'kmeans_prediction': 'spam' if kmeans_pred == 1 else 'not spam'
            })
    else:
        predict_form = PredictForm()
    return redirect('index')
