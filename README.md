# 📰 News Headline Classifier Deployment

## 1. Introduction

This repository contains the complete solution for a **News Headline Classification** system. The goal is to categorize news headlines into one of four categories: **Business (b)**, **Science/Tech (t)**, **Entertainment (e)**, or **Health (m)**.

The solution utilizes a highly efficient **Traditional Machine Learning (ML)** pipeline, leveraging **TF-IDF features** and a **Logistic Regression** classifier, which achieved high accuracy ($\approx 94.43\%$) while maintaining low computational overhead for deployment.

The application is containerized using **Docker** and orchestrated using **Kubernetes (K8s)** for robust, scalable serving.

## 2. Model and Architecture

### Model Choice: TF-IDF + Logistic Regression

Instead of a large deep learning model, I chose a traditional ML approach for efficiency:

| Component | Role | Description |
| :--- | :--- | :--- |
| **Data Preprocessing** | Upsampling | Minority classes (`b`, `t`, `m`) were upsampled to balance the dataset based on the size of the majority class (`e`). |
| **Feature Extraction** | **TfidfVectorizer** | Converts variable-length text headlines into a **fixed-length, sparse numerical vector**. This captures the importance of words by weighing their frequency in the document against their rarity across the entire dataset. |
| **Classifier** | **Logistic Regression** | A fast, linear model trained on the TF-IDF vectors. It provides high performance with minimal inference latency and memory requirements. |

### Serving Stack

* **API Framework:** **FastAPI** (`src/api.py`) for defining a high-performance REST endpoint.
* **Containerization:** **Docker** for packaging the Python environment, the saved model (`.pkl` files), and the FastAPI application.
* **Orchestration:** **Kubernetes (K8s)** for managing multiple replicas and providing a scalable, load-balanced entry point.

## 🛠️ 3. Setup and Prerequisites

You need the following tools installed and configured:

* **Python 3.8+**
* **Docker** (Installed and running)
* **kubectl** (Configured to connect to a Kubernetes cluster, e.g., MiniKube, GKE, EKS)
* **The Dataset:** Ensure the `newsCorpora.csv` file is placed in the `news aggregator folder/` directory.

## 📂 4. Project File Structure

```

tf-idf-deployment/
├── news aggregator folder/
│   └── newsCorpora.csv        \# Raw Data
├── requirements.txt           \# Python packages
├── Dockerfile                 \# Docker build file
├── .dockerignore              \# Docker ignore
├── k8s-deployment.yaml        \# K8s Deployment definition
├── k8s-service.yaml           \# K8s Service definition
├── src/
│   └── model.py               \# Training and saving script
│   └── api.py                 \# FastAPI serving logic
└── tfidf\_logreg\_classifier/   \# Model Artifacts (Generated after running src/model.py)
├── logreg\_model.pkl
└── tfidf\_vectorizer.pkl

````

---

## 🚀 5. Deployment Steps

Follow these steps sequentially to train the model, build the container, and deploy it to Kubernetes.

### Step 5.1: Train and Save the Model Artifacts

Run the training script to generate the necessary `.pkl` files.

1.  Navigate to the root directory (`tf-idf-deployment`).
2.  Install local dependencies (only needed to run the training script):
    ```bash
    pip install -r requirements.txt
    ```
3.  Execute the training script:
    ```bash
    python src/model.py
    ```
    * **Result:** This creates the `tfidf_logreg_classifier/` directory containing `logreg_model.pkl` and `tfidf_vectorizer.pkl`.

### Step 5.2: Build the Docker Image

Build the container image that packages the application and the saved model artifacts.

1.  Ensure Docker is running.
2.  Build the image, tagging it locally:
    ```bash
    docker build -t logreg-classifier-api:v1 .
    ```
3.  (Optional, but required for remote K8s clusters) Push the image to a remote registry (e.g., Docker Hub, GCR, ECR):
    ```bash
    # Replace 'your-registry/repo' with your actual path
    docker tag logreg-classifier-api:v1 your-registry/repo/logreg-classifier-api:v1
    docker push your-registry/repo/logreg-classifier-api:v1
    ```

### Step 5.3: Deploy to Kubernetes (K8s)

Apply the configuration files to your cluster. **Ensure the image path in `k8s-deployment.yaml` matches your pushed image tag.**

1.  **Deploy the Application:** Create the Deployment object, which ensures two replicas of the API are running.
    ```bash
    kubectl apply -f k8s-deployment.yaml
    ```
2.  **Deploy the Service:** Create the Service object, which exposes the deployment externally.
    ```bash
    kubectl apply -f k8s-service.yaml
    ```

### Step 5.4: Access the API Endpoint

1.  **Wait** for the LoadBalancer to provision an external IP (this can take a few minutes on cloud providers):
    ```bash
    kubectl get service distilbert-news-classifier-service
    ```
2.  Once the `EXTERNAL-IP` is available, you can test the `/predict` endpoint using curl or a tool like Postman:
    ```bash
    # Replace <EXTERNAL-IP> with the IP retrieved from the command above
    curl -X POST http://<EXTERNAL-IP>/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Apple launches new iPhone with better battery life."}'
    ```
    * **Expected Output:** A classification result for the headline (likely 't' for Science/Tech or 'b' for Business).
````