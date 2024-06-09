import re
import json
import numpy as np
import pandas as pd
from random import randint
from math import sqrt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render


def index(request):
    return render(request, 'main/predictor.html')


# Utility functions

def extract_features(email_text):
    words = re.findall(r'\b\w+\b', email_text.lower())
    return Counter(words)

def load_data(file):
    df = pd.read_csv(file)
    email_texts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].apply(lambda x: 1 if x == '1' else 0).tolist()
    features = [extract_features(email) for email in email_texts]
    vectorized_data = pd.DataFrame(features).fillna(0).values
    return email_texts, vectorized_data, labels

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point2)):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)

def k_means(data, k):
    centers = [data[randint(0, len(data) - 1)] for _ in range(k)]
    labels = [-1] * len(data)
    max_iterations = 700

    for _ in range(max_iterations):
        new_labels = []
        for point in data:
            distances = [euclidean_distance(point, center) for center in centers]
            new_labels.append(np.argmin(distances))

        if new_labels == labels:
            break
        labels = new_labels

        new_centers = np.zeros_like(centers)
        counts = np.zeros(k)
        for label, point in zip(labels, data):
            new_centers[label] += point
            counts[label] += 1
        centers = [new_centers[i] / counts[i] if counts[i] > 0 else centers[i] for i in range(k)]

    return labels, centers

def predict_knn(train_data, train_labels, test_point, k):
    distances = []
    for i, train_point in enumerate(train_data):
        distance = euclidean_distance(train_point, test_point)
        distances.append((distance, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]


# Django views

@csrf_exempt
def train(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        email_texts, data, labels = load_data(file)
        
        global train_data, train_labels, kmeans_labels, centers
        train_data, train_labels = data, labels
        k = 2
        kmeans_labels, centers = k_means(data, k)
        print(kmeans_labels, centers)
        return JsonResponse({'message': 'Training completed successfully.'})
    return JsonResponse({'error': 'Invalid request.'}, status=400)


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        email_text = request.POST.get('email_text', '')
        k = int(request.POST.get('k', 5))
        
        if not email_text:
            return JsonResponse({'error': 'No email text provided.'}, status=400)

        test_data = extract_features(email_text)
        vectorized_data = pd.DataFrame([test_data]).fillna(0).values[0]
        
        global train_data, kmeans_labels
        predicted_label = predict_knn(train_data, kmeans_labels, vectorized_data, k)
        
        return JsonResponse({'predicted_label': 'spam' if predicted_label == 1 else 'not spam'})
    return JsonResponse({'error': 'Invalid request.'}, status=400)
