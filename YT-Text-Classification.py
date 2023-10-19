##############################################################################################################

''' This code serves the purpose of gathering YouTube comments related to news outlets covering the war in 
Ukraine and classifying those comments according to the respective news outlets. 
It combines web scraping techniques using the YouTube API with machine learning for comment classification. '''

##############################################################################################################

# Install required libraries using pip
# pip install google-api-python-client nltk tensorflow

# Import required libraries
import json
import pandas as pd
import googleapiclient.discovery
import random
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.probability import FreqDist
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to scrape comments from a YouTube video
def scrape_video_comments(video_id, max_results=100):
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=max_results
    )

    while request:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        request = youtube.commentThreads().list_next(request, response)

    return comments

# Set up API credentials 
api_key = 'your_api_key_here'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

# List of video IDs
video_ids = ['Y-ZaFDWyZ38', 'Av0Fkw1GAZE', 'xYGtxtYo6lg', 'l7C7vNa2-K0', 'kyaALHVnEW8', 'mPbkdmYkGjQ', 'JB57sf112So', 'aux9IijLYoc', '9KNUwmOWC5I', 'z1riW7za2EQ', 'ahNkBiB1Q7g', 'FmnQDnNS2qw']
# Fox News (3), CNN (3), BBC (3), Sky News (3)

# Create a dictionary to store comments
all_comments = {}
for video_id in video_ids:
    comments = scrape_video_comments(video_id)
    all_comments[video_id] = comments

# Convert the dictionary to a DataFrame
df = pd.DataFrame(columns=['VideoID', 'Comments'])
for video_id, comments in all_comments.items():
    df = df.append({'VideoID': video_id, 'Comments': comments}, ignore_index=True)

##############################################################################################################

''' Format data set '''

##############################################################################################################

# Labels for the videos by news outlet (Fox News, CNN, BBC, Sky News)
df['Labels'] = ['FoxNews', 'FoxNews', 'FoxNews', 'CNN', 'CNN', 'CNN', 'BBC', 'BBC', 'BBC', 'SkyNews', 'SkyNews', 'SkyNews']

# Explode the list column to separate comments
df_explode = df.explode('Comments', ignore_index=True)

# Inspect if the distribution of comments between news outlets is roughly even

# Display the distribution of comments between news outlets
video_counts = df_explode['Labels'].value_counts()
print(video_counts)

# Find which videos have the most comments

# Display the distribution of comments between videos
video_counts = df_explode['VideoID'].value_counts()
print(video_counts)

# Randomly delete records from two of the CNN videos to balance the dataset
start_range_1, end_range_1 = 1225, 2650
start_range_2, end_range_2 = 2651, 4619
num_rows_to_delete_1 = min(800, end_range_1 - start_range_1)
num_rows_to_delete_2 = min(1200, end_range_2 - start_range_2)
indices_to_delete_1 = list(range(start_range_1, end_range_1))
random.shuffle(indices_to_delete_1)
indices_to_delete_2 = list(range(start_range_2, end_range_2))
random.shuffle(indices_to_delete_2)
rows_to_delete_1 = indices_to_delete_1[:num_rows_to_delete_1]
rows_to_delete_2 = indices_to_delete_2[:num_rows_to_delete_2]
df_even_records = df_explode.drop(rows_to_delete_1 + rows_to_delete_2)
df_even_records.reset_index(drop=True, inplace=True)

# Verify change

# Use value_counts() to count records in each video
video_counts = df_even_records['Labels'].value_counts()

# Display the counts
print(video_counts)

##############################################################################################################

''' Clean text data '''

##############################################################################################################

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')

''' The following three functions are adopted from the 'text_classification_glove_embeddings.ipynb' notebook 
by embedded-robotics (https://github.com/embedded-robotics/datascience)'''

def get_wordnet_pos(tag):
    ''' Map part-of-speech tags to WordNet POS tags for lemmatization. This function converts Penn Treebank 
    POS tags (e.g., 'JJ', 'VB', 'NN', 'RB') to WordNet POS tags. '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(word_list):
    ''' Lemmatize a list of words using WordNetLemmatizer. This function takes a list of words, 
    performs part-of-speech tagging, and then lemmatizes each word based on its POS tag using WordNetLemmatizer.
     It returns the lemmatized words as a space-separated string. '''
    wl = WordNetLemmatizer()
    word_pos_tags = pos_tag(word_list)
    lemmatized_list = []
    for tag in word_pos_tags:
        lemmatize_word = wl.lemmatize(tag[0], get_wordnet_pos(tag[1]))
        lemmatized_list.append(lemmatize_word)
    return " ".join(lemmatized_list)

def clean_text(text):
    ''' Clean and preprocess the input text.'''
    text = str(text).strip()
    text = str(text).lower()
    text = re.sub(r"\n", r" ", text)
    word_tokens = word_tokenize(text)
    cleaned_text = []
    for word in word_tokens:
        cleaned_text.append("".join([char for char in word if char.isalnum()]))
    stop_words = stopwords.words('english')
    text_tokens = [word for word in cleaned_text if (len(word) > 2) and (word not in stop_words)]
    text = lemmatize(text_tokens)
    return text

''' DataFrame with cleaned comments stored as strings '''

# Apply text cleaning function to comments
comments_col = df_even_records['Comments']
comments_clean = comments_col.apply(lambda x: clean_text(x))

# Store clean text into DataFrame with comments stored as strings
df_string = {'Comments': comments_clean, 'Labels': df_even_records['Labels']}
df_string = pd.DataFrame(df_string)

# Store comments as lists of words
comments_clean_list = comments_clean.str.split(' ')

''' DataFrame with cleaned comments stored as lists of words '''

# Store clean text into DataFrame with comments stored as lists
df_list = {'Comments': comments_clean_list, 'Labels': df_even_records['Labels']}
df_list = pd.DataFrame(df_list)

##############################################################################################################

''' Exploratory analysis '''

##############################################################################################################

# Sample categorical data
categories = list(df_list['Labels'])

# Calculate the frequency of each category using Counter
category_counts = Counter(categories)

# Extract category labels and their frequencies
labels = list(category_counts.keys())
frequencies = list(category_counts.values())

# Create a bar plot (histogram) for categorical data
plt.bar(labels, frequencies)
plt.title('Histogram of Comments by News Outlet')
plt.xlabel('News Outlet')
plt.ylabel('Frequency')
plt.show()

# Calculate the word count for each text from raw data
word_counts = [len(text.split()) for text in df_even_records['Comments']]

# Calculate the total word count
total_word_count = sum(word_counts)

# Calculate the average word count
average_word_count = total_word_count / len(df_even_records['Comments'])

print("Total word count of raw data:", total_word_count)
print("Average word count of raw data:", average_word_count)

# Calculate the word count for each text from cleaned data
word_counts = [len(text.split()) for text in comments_clean]

# Calculate the total word count
total_word_count = sum(word_counts)

# Calculate the average word count
average_word_count = total_word_count / len(comments_clean)

print("Total word count from cleaned data:", total_word_count)
print("Average word count from cleaned data:", average_word_count)

# Calculate the word count for each text from raw data by category ('Labels')
word_counts_by_category = df_even_records.groupby('Labels')['Comments'].apply(lambda x: x.str.split().apply(len).mean()).reset_index()
word_counts_by_category.rename(columns={'Comments': 'Average Word Count'}, inplace=True)

# Calculate the total word count by category
total_word_count_by_category = df_even_records.groupby('Labels')['Comments'].apply(lambda x: x.str.split().apply(len).sum()).reset_index()
total_word_count_by_category.rename(columns={'Comments': 'Total Word Count'}, inplace=True)

print("Word counts by category for raw data:")
print(word_counts_by_category)
print("\nTotal word count by category for raw data:")
print(total_word_count_by_category)

# Calculate the word count for each text from raw data by category ('Labels')
word_counts_by_category = df_string.groupby('Labels')['Comments'].apply(lambda x: x.str.split().apply(len).mean()).reset_index()
word_counts_by_category.rename(columns={'Comments': 'Average Word Count'}, inplace=True)

# Calculate the total word count by category
total_word_count_by_category = df_string.groupby('Labels')['Comments'].apply(lambda x: x.str.split().apply(len).sum()).reset_index()
total_word_count_by_category.rename(columns={'Comments': 'Total Word Count'}, inplace=True)

print("Word counts by category for raw data:")
print(word_counts_by_category)

print("\nTotal word count by category for raw data:")
print(total_word_count_by_category)

''' Inspect frequency of most common words. '''

# Store text as one list of words
single_list = [item for sublist in comments_clean_list for item in sublist]

# Frequency Distribution
fdist = FreqDist(single_list)

# Print most common words
print("Most common words:")
for word, frequency in fdist.most_common(30):
    ''' Here we print the 30 most common words '''
    print(f"{word}: {frequency}")

# Bar plot of word frequency
word_freq_df = pd.DataFrame(fdist.most_common(30), columns=['Word', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=word_freq_df)
plt.title('Top 30 Most Common Words')
plt.show()

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(single_list))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# List of words to search for
search_words = list(word_freq_df['Word'])

# Function to count occurrences of search words in a list of words
def count_occurrences(word_list):
    word_count = Counter(word_list)
    return {word: word_count[word] for word in search_words}

# Apply the function to the 'Comments' column and store results in a new column
df_list['WordCounts'] = df_list['Comments'].apply(count_occurrences)

# Initialize a dictionary to store total counts for each news outlet
total_counts_dict = {}

# Calculate the total non-zero counts for each search word for each news outlet
for category in df_list['Labels'].unique():
    category_data = df_list[df_list['Labels'] == category]
    category_total_counts = {word: category_data['WordCounts'].apply(lambda x: x[word] if word in x else 0).sum() for word in search_words}
    total_counts_dict[category] = category_total_counts

# Convert the total counts dictionary to a DataFrame
total_counts_df = pd.DataFrame(total_counts_dict)

# Create a stacked bar plot
ax = total_counts_df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Top 30 Words Count by News Outlet")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.legend()
plt.show()


''' TF-IDF scores of unigrams and bigrams

We use code from https://www.analyticsvidhya.com/blog/2021/11/a-guide-to-building-an-end-to-end-multiclass-text-classification-model
to implement the TF-IDF vectorizer and analyze text data correlations. '''


# Create a new column 'CategoryID' with encoded categories
df_string['CategoryID'] = df_string['Labels'].factorize()[0]
category_id_df = df_string[['Labels', 'CategoryID']].drop_duplicates()

# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['CategoryID', 'Labels']].values)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')

# Transform text data into TF-IDF features
features = tfidf.fit_transform(df_string.Comments).toarray()
labels = df_string.CategoryID

# Print the shape of the features matrix
print("Shape of features matrix:", features.shape)

# Calculate the most correlated unigrams and bigrams with each news outlet
N = 5
for news, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("News Outlet: %s" % news)
    print("  * Most Correlated Unigrams: %s" % ', '.join(unigrams[-N:]))
    print("  * Most Correlated Bigrams: %s" % ', '.join(bigrams[-N:]))

##############################################################################################################

    ''' Classification '''
    
##############################################################################################################

''' We utilize a set of classifiers including RandomForestClassifier,
LinearSVC, MultinomialNB, and LogisticRegression, along with 5-fold cross-validation. '''

# Split the data into training (75%) and test sets (25%)
X = df_list['Comments']
y = df_list['Labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define a set of classifiers
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(multi_class='multinomial', random_state=0),
]

# Perform 5-fold cross-validation for each classifier
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Calculate and display the mean accuracy and standard deviation for each classifier
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis=1, ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard Deviation']

# Display summary of accuracies
print(acc)

# Plot a boxplot of accuracy for each classifier
plt.figure(figsize=(8, 5))
sns.boxplot(x='model_name', y='accuracy', data=cv_df, color='lightblue', showmeans=True)
plt.title("Mean Accuracy (CV = 5)")
plt.xlabel("Model Name")
plt.ylabel("Accuracy")
plt.show()

''' We train a simple Multinomial NB classification model with cross-fold-validation 
and view results in a confusion matrix. '''

# Split the data into training (75%) and test sets (25%)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df_list.index, test_size=0.25, random_state=1)

# Define the Multinomial NB model
model = MultinomialNB(multi_class='multinomial', random_state=0)

# Perform 5-fold cross-validation to assess model performance
CV = 5
accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

# Print the mean accuracy and standard deviation from cross-validation
print(f"Mean Accuracy: {accuracies.mean():.2f}")
print(f"Standard Deviation: {accuracies.std():.2f}")

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print classification report
classification_report = metrics.classification_report(y_test, y_pred, target_names=df_list['Labels'].unique())
print(classification_report)

# Generate a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=category_id_df.Labels.values,
            yticklabels=category_id_df.Labels.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Multinomial NB", size=16)

''' We implement a Convolutional Neural Network (CNN) for text classification.'''

# Store comments and their CategoryID
df_string_ID = df_string[['Comments','CategoryID']]
df_string_ID.head() # Verify change

# Separate data into training (75%) and test sets (25%)
X = df_string_ID['Comments'] # Collection of comments
y = df_string_ID['CategoryID'] # Labels we want to predict 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

# Initialize and fit a tokenizer on the training data
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(X_train)

# Get the word-to-index mapping
tokenizer.index_word

# Calculate the vocabulary size
vocab_size = len(tokenizer.index_word) + 1
print('Vocabulary Size:', vocab_size)

# Convert text data to sequences of integers
X_train_token = tokenizer.texts_to_sequences(X_train)
X_test_token = tokenizer.texts_to_sequences(X_test)

# Set a fixed sequence length for padding
sequence_len = 300

# Pad sequences to have the same length
X_train_token = pad_sequences(X_train_token, padding='post', maxlen=sequence_len)
X_test_token = pad_sequences(X_test_token, padding='post', maxlen=sequence_len)

# Define the embedding dimension
embedding_dim = 100

# Create a Sequential model
model = Sequential()

# Add an Embedding layer for word embeddings
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_len))

# Add a 1D Convolutional layer
model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))

# Add a Global Max Pooling layer
model.add(layers.GlobalMaxPool1D())

# Add Dense layers for classification
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_token, y_train, epochs=20, validation_data=(X_test_token, y_test), batch_size=128)

# Create a DataFrame to store training metrics
metrics_df = pd.DataFrame(history.history)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(metrics_df.index, metrics_df.loss)
plt.plot(metrics_df.index, metrics_df.val_loss)
plt.title('CNN Training with Word Embeddings')
plt.xlabel('Epochs')
plt.ylabel('Categorical Crossentropy Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(metrics_df.index, metrics_df.accuracy)
plt.plot(metrics_df.index, metrics_df.val_accuracy)
plt.title('CNN Training with Word Embeddings')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()
