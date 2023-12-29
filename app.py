import re
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from streamlit_shap import st_shap

# Function Definitions
# ------------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Clean the news titles and remove stopwords
    :param text: news title (uncleaned)
    :return: cleaned text
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


@st.cache_data
def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load and preprocess the data
    :return: preprocessed dataframe
    """
    df = pd.read_csv('data/all-data.csv', encoding='latin_1', header=None, names=["label", "text"])
    df.drop_duplicates(inplace=True)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


@st.cache_resource
def train_model(data: pd.DataFrame) -> tuple:
    """
    Train the model
    :param data: preprocessed dataframe
    :return: model, vectorizer, X_test, y_test
    """
    X = data["clean_text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
    logistic_regression = LogisticRegression(C=1.5)
    logistic_regression.fit(X_train_tfidf, y_train)
    return logistic_regression, tfidf_vectorizer, X_test, y_test


@st.cache_data
def generate_dataset_for_explainer(X_test: pd.Series, y_test: pd.Series) -> tuple:
    """
    Generate dataset for shap explainer
    :param X_test: cleaned news title test dataset
    :param y_test: label test dataset
    :return: test datasets with reset index
    """
    X_test_reset = X_test.reset_index()  # Keep the original index to get original text
    X_test_reset.columns = ['original_index', 'clean_text']  # Rename columns for clarity
    y_test_reset = y_test.reset_index(drop=True)
    return X_test_reset, y_test_reset


@st.cache_data
def get_shap_values(_model: LogisticRegression, _vectorizer: TfidfVectorizer, _X_test: pd.Series) -> shap.Explanation:
    """
    Get shap values
    :param _model: LogisticRegression model
    :param _vectorizer: TfidfVectorizer
    :param _X_test: X_test dataset
    :return: shap values
    """
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    explainer = shap.Explainer(model, X_test_tfidf, feature_names=vectorizer.get_feature_names_out())
    shap_values = explainer(X_test_tfidf)
    return shap_values


def explain_single_predicion(model, vectorizer, df, X_test_reset, y_test_reset, shap_values, label):
    filtered_y_test = y_test_reset[y_test_reset == label]
    if not filtered_y_test.empty:
        random_sample = filtered_y_test.sample()
        random_idx = random_sample.index.to_numpy()
        random_true_label = random_sample.values[0]
        pred_label = model.predict(vectorizer.transform([X_test_reset.iloc[random_idx]["clean_text"].values[0]]))[0]

        st.subheader("Prediction")
        # st.divider()
        original_idx = X_test_reset.iloc[random_idx]["original_index"]  # Use iloc
        text = df.loc[original_idx]["text"].item()
        st.markdown(f'''**News Title**: {text}
                ''')

        if random_true_label == pred_label:
            st.markdown(f'''
            **True Label**: :green[{random_true_label}]  
            **Predicted Label**: :green[{pred_label}]
            ''')
        else:
            st.markdown(f'''
            **True Label**: :red[{random_true_label}]  
            **Predicted Label**: :red[{pred_label}]
            ''')

        st.markdown("**Negative**")
        st_shap(shap.plots.force(shap_values[random_idx, :, 0]))

        st.markdown("**Neutral**")
        st_shap(shap.plots.force(shap_values[random_idx, :, 1]))

        st.markdown("**Positive**")
        st_shap(shap.plots.force(shap_values[random_idx, :, 2]))


# Main App Flow
# -----------------------------------------------------------------------------

# Setup page title and header
st.title('Unveiling Sentiment Analysis: A SHAP-Driven Explanation')
st.header('Build a Financial News Sentiment Classification Model using Logistic Regression')

# Global stopwords set
stop_words = set(stopwords.words('english'))

# Data loading and preprocessing
st.subheader('Dataset')
st.write('''
The dataset (FinancialPhraseBank) contains the sentiments for financial news headlines from the perspective of a retail investor.  
Available on Kaggle: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data  

The data shown below was cleaned and preprocessed.
''')
df = load_and_preprocess_data()
st.dataframe(df)

# Show clean text function
if st.checkbox('Click here to view source code'):
    st.code('''
    def clean_text(text: str) -> str:
        """
        Clean the news titles and remove stopwords
        :param text: news title (uncleaned)
        :return: cleaned text
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return " ".join(filtered_tokens)
    ''')

# TF-IDF Vectorization and Train/Test Split
st.subheader('TF-IDF Vectorization and Train/Test Split')
st.write('Extract features from the text using TF-IDF Vectorizer and split the data into train and test sets.')
st.code('''
# Split the data into features (X) and labels (y)
X = df["clean_text"]
y = df["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer and transform the text data
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()
''')

# Train the model and prepare data for SHAP
model, vectorizer, X_test, y_test = train_model(df)
X_test_reset, y_test_reset = generate_dataset_for_explainer(X_test, y_test)

# Model Evaluation
st.subheader('Model Evaluation')
if st.button('View model evaluation'):
    y_pred = model.predict(vectorizer.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    # Create two columns for the confusion matrix and the report
    col1, col2 = st.columns(2)

    # Display confusion matrix in the first column
    with col1:
        st.markdown('**Confusion Matrix**')
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(values_format='.0f', cmap='Blues', ax=ax)
        ax.grid(False)
        st.pyplot(fig)

    # Display classification report in the second column
    with col2:
        st.write('**Classification Report**')
        st.dataframe(report)

# SHAP Explanation Section
st.header("Explain the Logistic Regression Model with SHAP")
st.markdown('''
SHAP (SHapley Additive exPlanations) is a tool in machine learning that explains the output of any machine learning model. It's based on Shapley values from cooperative game theory and offers the following features:

1. **Model Agnostic**: Works with any machine learning model.
2. **Local and Global Explanations**: Provides detailed explanations for individual predictions and overall model behavior.
3. **Feature Importance**: Shows how much each feature in the dataset contributes to a prediction.
4. **Intuitive and Fair**: The explanations are easy to understand and consistently represent the impact of each feature.

Essentially, SHAP helps in understanding why a model makes certain predictions, making machine learning models more transparent and interpretable.
''')
shap_values = get_shap_values(model, vectorizer, X_test)

# beaswarm plot
st.subheader("Visualize the Impact of Features for Each Label")
label_selection = st.selectbox('Select a label to visualize', ('negative', 'positive', 'neutral'))
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
max_display = st.slider("Maximum number of features to display", min_value=1, max_value=20, value=10)
if label_selection:
    st.markdown(f"##### {label_selection}")
    st_shap(shap.plots.beeswarm(shap_values[:, :, label_dict[label_selection]], max_display=max_display))

# single prediction
st.subheader("See how features contributed to the model’s prediction for each label.")
option = st.selectbox('Select a label to generate a prediction', ('negative', 'positive', 'neutral'))
if st.button("Generate a prediction", type="primary"):
    explain_single_predicion(model, vectorizer, df, X_test_reset, y_test_reset, shap_values, option)
