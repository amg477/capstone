import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text, remove_stopwords=True):
    """
    Cleans raw text by removing URLs, punctuation, numbers, mentions,
    and optionally stopwords.
    
    Args:
        text (str): The input text string.
        remove_stopwords (bool): Whether to remove stopwords (default: True).
    
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    cleaned_text = " ".join(tokens)
    
    return cleaned_text