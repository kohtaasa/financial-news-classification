# Financial News Sentiment Analysis with SHAP

## Introduction
This project is a Streamlit-based web application for sentiment analysis on financial news. It leverages Logistic Regression to classify news into sentiments like positive, negative, or neutral. The application also uses **SHAP** (SHapley Additive exPlanations) to provide interpretable insights into the model's predictions.

## Features
- Sentiment classification of financial news using Logistic Regression.
- Text data preprocessing including stopword removal.
- TF-IDF vectorization.
- Model evaluation with confusion matrix and classification report.
- SHAP analysis for model interpretability.
- Streamlit interface for interactive user experience.

## Usage
- The interactive [app](https://sentiment-shap.streamlit.app/) is built and hosted with Streamlit.
- Detailed analysis is available in the `sentiment_analysis.ipynb` notebook.

## Getting Started

### Simple installation
- Install the dependencies with `poetry install --without expt`
- Run the app with `streamlit run app.py`
- Open the app in your browser with `http://localhost:8501`
### Using docker
- Build the image with `docker build -t sentiment-shap .`
- Run the app with `docker run -p 8501:8501 sentiment-shap`
- Open the app in your browser with `http://localhost:8501`