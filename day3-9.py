import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string
import nltk

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Function to plot bar graph
def plot_bar_graph(words, freqs, n):
    plt.bar(words[:n], freqs[:n])
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {n} Most Frequent Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Load dataset from CSV
def load_dataset(filename):
    df = pd.read_csv(filename)
    return df['FEEDBACK']

# Main function
def main():
    # Load dataset
    dataset = load_dataset('DAY3_9.csv')

    # Preprocess text
    tokenized_text = dataset.apply(preprocess_text).sum()

    # Calculate frequency distribution
    fdist = FreqDist(tokenized_text)

    # Get user input for top N words
    n = int(input("Enter the number of top words you want to analyze: "))

    # Get top N words and their frequencies
    top_words = [word for word, freq in fdist.most_common(n)]
    top_freqs = [freq for word, freq in fdist.most_common(n)]

    # Display top N words and their frequencies
    for word, freq in zip(top_words, top_freqs):
        print(f'{word}: {freq}')

    # Plot bar graph
    plot_bar_graph(top_words, top_freqs, n)

if __name__ == "__main__":
    main()
