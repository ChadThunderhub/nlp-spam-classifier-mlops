# NLP Spam Classifier | MLOps Pipeline

A production-ready Natural Language Processing (NLP) engine for automated threat detection (Spam/Phishing), fully containerized and deployed using modern MLOps practices.

## Project Overview
This repository showcases an end-to-end Machine Learning lifecycle. It starts with raw data analysis and model training in Jupyter Notebooks, culminating in a fully containerized Streamlit web application. The system leverages Scikit-Learn and NLTK for high-accuracy binary text classification.

## Tech Stack & Architecture
* **Machine Learning:** Scikit-Learn (SVM, Random Forest, Naive Bayes), Pandas, NumPy
* **NLP Processing:** NLTK (WordNet Lemmatization, TF-IDF Vectorization)
* **Frontend/Serving:** Streamlit
* **Infrastructure:** Docker, Docker Compose
* **CI/CD:** GitHub Actions (Automated build and health checks)

## Key Features
* **Active Payload Scanner:** Real-time text classification with probability metrics.
* **Tuned SVM Engine:** Hyperparameter-optimized Support Vector Machine yielding peak F1-Score.
* **Visual Analytics Dashboard:** Interactive confusion matrices, feature importance graphs, and semantic word clouds.
* **Immutable Infrastructure:** Multi-stage Docker build with built-in health checks.

## Quickstart (Local Deployment)

### Prerequisites
* [Docker](https://www.docker.com/) and Docker Compose installed.

### Spin up the environment
1. Clone the repository:
   ```bash
   git clone https://github.com/ChadThunderhub/nlp-spam-classifier-mlops.git
   cd nlp-spam-classifier-mlops
   ```

2. Build and start the container in detached mode:
   ```bash
   docker-compose up --build -d
   ```

4. Access the application at: http://localhost:8501

To stop the application, run:
   ```bash
   docker-compose down
   ```
