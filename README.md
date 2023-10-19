# YouTube Comment Analysis and Classification

In the following, we collect YouTube comments associated with news outlets reporting on the conflict in Ukraine. These comments are then categorized based on their respective news sources. The project leverages a combination of a YouTube API for web scraping and of machine learning algorithms for comment classification.

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/de614fc4-458a-467b-aeec-f8fac94aa537)

## Research Objective

The objective of this project is to determine whether we can effectively classify YouTube comments originating from four distinct news sources based on their content. We have specifically opted to analyze comments from the following news outlets: Fox News, CNN, BBC, and Sky News. These selections were made because they offer diverse political perspectives from two different countries while also having overlapping viewership.

Fox News is an American multinational conservative news and political commentary channel headquartered in New York City. It is often noted for its partisan reporting in favour of the Republican Party, often portraying the Democratic Party in a negative light.

CNN, the Cable News Network, is a multinational news channel and website based in Atlanta, Georgia. It is renowned for its live and dramatic coverage of breaking news, although some critics have accused it of being overly sensationalistic. Additionally, it has been suggested that CNN displays a bias in favour of the Democratic Party.

The British Broadcasting Corporation (BBC) is a British public service broadcaster with its headquarters at Broadcasting House in London. Over time, the BBC has faced allegations of liberal and left-wing bias.

Sky News is a British free-to-air television news channel and organization situated in London. There have been claims of inherent bias, particularly during the 1990s and 2000s when it was minority-owned and dominated by Rupert Murdoch's right-leaning News Corporation. Subsequently, it fell under the control of the Murdoch family's 21st Century Fox.

To maintain consistency in our data, we have chosen three videos from each of these sources, all related to the Ukraine-Russia conflict and published between July and September 2023. This topic was selected because it is likely that regardless of political affiliation, viewers of these videos will have relatively similar overall sentiment about the conflict. This approach allows us to control as many variables as possible in our analysis.

## Methods

We use an API to scrape YouTube comments and organize them into a data frame alongside labels indicating their respective sources. To clean the YouTube comment text data, we utilize the Python package NLTK and incorporate functions from the 'text_classification_glove_embeddings.ipynb' notebook authored by embedded-robotics. You can access this notebook in their [GitHub repository](https://github.com/embedded-robotics/datascience). These functions play a pivotal role in effective text data preprocessing:

- `get_wordnet_pos`: Maps Penn Treebank part-of-speech tags to WordNet part-of-speech (POS) tags.
- `lemmatize`: Performs lemmatization based on POS tags.
- `clean_text`: Carries out extensive text cleaning, including tasks such as lowercasing, tokenization, removal of non-alphanumeric characters, stopwords, and lemmatization.

Subsequently, we conduct an exploratory analysis of the data, focusing on the distribution of comments per source, the total word count, and average word count. Additionally, we implement the TF-IDF vectorizer and conduct text data correlation analysis using code from [Analytics Vidhya's guide](https://www.analyticsvidhya.com/blog/2021/11/a-guide-to-building-an-end-to-end-multiclass-text-classification-model). Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used in information retrieval and text mining to represent the importance of a term within a document relative to a collection of documents. The TF-IDF value is calculated by multiplying the the frequency of a term and the uniqueness or rarity of a term across the entire document. When you use a TF-IDF vectorizer to convert a collection of text documents into a numerical feature matrix, the shape of the resulting matrix is determined by the number of documents (rows) and the number of unique terms (words) in the corpus (columns).

Our classification process involves comparing machine learning algorithms, including Random Forest, Linear Support Vector Classifier (LinearSVC), Multinomial Naive Bayes (MultinomialNB), and Logistic Regression, configured with the 'multinomial' option. Finally, we aim to enhance these results by implementing a convolutional neural network (Convolutional-NN, distinct from CNN, the news outlet) with word embeddings. Here is a brief description of each technique:

**Random Forest:** Random Forest is composed of a collection of decision trees. A decision tree is a tree-like structure that partitions the data into subsets based on features and their values. Each leaf node of the tree represents a class or a numerical value. Decision trees are simple, interpretable models that can capture complex relationships in data.

**Linear SVC:** Linear SVC is a machine learning algorithm used for classification tasks. It belongs to the family of Support Vector Machines (SVMs) and, specifically, is a variation of SVM designed for linear classification problems. It is primarily used for binary and multiclass classification, where the goal is to separate data points into different classes or categories based on their features.

**Multinomial NB:** Multinomial NB is a probabilistic classification algorithm commonly used for text classification and other categorical data classification tasks. It is part of the Naive Bayes family of algorithms, which are based on Bayes' theorem and the assumption of conditional independence between variables.

**Logistic Regression:** LogisticRegression is a popular classification algorithm used for binary and multiclass classification problems. It models the probability of belonging to a class using a logistic function. In this case, it's configured for multiclass classification using the 'multinomial' option. It extends the binary logistic regression to handle situations where there are more than two distinct classes to predict.

**Convolutional-NN:** A Convolutional-NN is a type of artificial neural network designed for processing structured grid data, such as images and, to some extent, sequential data like time series or text. They use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. These layers consist of filters or kernels, which are small matrices that slide (convolve) over the input data. The filters detect specific patterns or features in the data, such as edges, corners, and textures. Word embeddings are distributed representations of words in a vector space, typically in a continuous numerical format. 


## Data Preparation

We utilized the YouTube API to gather comments from four prominent news sources: Fox News, CNN, BBC, and Sky News. We generate a data frame in the following format: 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VideoID</th>
      <th>Comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Y-ZaFDWyZ38</td>
      <td>[... In 1956 they invaded Hungary - in 1968 th...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Av0Fkw1GAZE</td>
      <td>[Russia is actively trying to gain voices in c...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xYGtxtYo6lg</td>
      <td>[All talk and no action. This is what politici...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>l7C7vNa2-K0</td>
      <td>[no wonder 🤔 u.s lose war in afgan, vietnam, s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kyaALHVnEW8</td>
      <td>[Amidst the usual propaganda, a little truth s...</td>
    </tr>
  </tbody>
</table>
</div>

We apply the relevant labels to these videos and examine the distribution of comments across different news sources. Notably, CNN stands out with a significantly higher comment count compared to the other sources, largely driven by two particular videos. To balance this distribution, we randomly remove 800 comments from one of the CNN videos and 1200 from the other. This adjustment results in a more uniform distribution; however, it is worth mentioning that Fox News still maintains the lowest comment count, trailing by around 500-600 comments compared to the other sources.

| News Source | Comment Count (Before) | Comment Count (After) |
|-------------|---------------|----------------|
| CNN         | 3811          | 1811          |
| BBC         | 1826          | 1826         |
| Sky News     | 1775          | 1775          |
| Fox News     | 1220          | 1220           |

In this project, we make use of the Python package NLTK and adopt functions from the 'text_classification_glove_embeddings.ipynb' notebook, found [here](https://github.com/embedded-robotics/datascience), to process our text data from the YouTube comments. We store the processed text in two separate data frames along with their corresponding labels. This allows for easy access to both data formats in the future. In one data frame, the comments are stored as strings, while in the other, the comments are stored as lists of individual words. Here is what the latter data frame structure looks like:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Comments</th>
      <th>Labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1956, invade, hungary, 1968, invaded, czechos...</td>
      <td>FoxNews</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[fox, love, criticize, dear, leader, control]</td>
      <td>FoxNews</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[provoke, pollute, romanian, navy, military, e...</td>
      <td>FoxNews</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[real, axis, evil]</td>
      <td>FoxNews</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[load, bull]</td>
      <td>FoxNews</td>
    </tr>
  </tbody>
</table>
</div>

## Exploratory Data Analysis

Before applying any classification techniques, we will perform an exploratory analysis of our dataset. Following the adjustments made earlier, we observe that the number of comments collected from the three videos per news source now exhibits a roughly the same number of comments across all four sources.

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/b5a4873c-e2cf-4cbd-b922-03e002ce9d88)


We conducted an analysis of the word count statistics, both before and after text data cleaning. We see an approximately 46.76% decrease in the total word count from the raw text data to the processed text data, and this is also reflected in the average word count per comment.

|                                 | Raw Data     | Processed Data  |
|---------------------------------------|--------------|---------------|
| Total Word Count                       | 187,841      | 100,077       |
| Average Word Count per Comment         | 28.32      | 15.09       |

Upon closer examination, we observed that the average word count for comments from BBC and CNN remained slightly above the overall average both before and after data cleaning. Conversely, for comments from Fox News and Sky News, the average word count was slightly below the overall average, both before and after the cleaning process.

|   | Average Word Count (Raw Data) | Average Word Count (Processed Data) |
|---------|-----------------------------|-----------------------------|
| BBC     | 30.68                       | 16.53                       |
| CNN     | 31.94                       | 17.02                       |
| Fox News | 25.11                       | 13.00                       |
| Sky News | 24.41                       | 13.08                       |

The following plot illustrates the top 30 most frequently used words, categorized by their respective news sources. Unsurprisingly, 'Ukraine,' 'Russia,' and 'war' emerge as the three most prominently featured words across all sources. Notably, our analysis reveals distinctive word usage patterns unique to each news source. For instance, 'tank' is frequent in content from Sky News but absent from Fox News. Conversely, 'Biden' is frequent in Fox News content, suggesting that these variations may be attributed to the distinct subject matter and content focus of each video.

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/574162f0-0093-4220-8254-511965c7e280)

We use the TF-IDF scores to analyze the correlation between unigrams (single words) and bigrams (pairs of words) within a set of comments. In our feature matrix, each row represents a video comment, and each column represents a unique word, where we have 6632 comments and 3620 features. The following results pertain to the top five most correlated unigrams and bigrams:

**BBC**
  * Most Correlated Unigrams: nazi, tank, cannafarm, succeed, bbc
  * Most Correlated Bigrams: seriously mention, slava russia, bbc propaganda, game changer, war crime
    
**CNN**
  * Most Correlated Unigrams: bbc, sober, chris, cnn, christie
  * Most Correlated Bigrams: cnn propaganda, news network, cnn say, dragon teeth, chris christie
    
**Fox News**
  * Most Correlated Unigrams: fox, biden, joe, impeach, mccarthy
  * Most Correlated Bigrams: impeach biden, talk action, talk talk, joe biden, fox news
    
**Sky News**
  * Most Correlated Unigrams: sky, changer, leopard, abrams, tank
  * Most Correlated Bigrams: abrams tank, sean bell, deplete uranium, sky news, game changer

In the case of all three news sources, the most correlated unigrams and bigrams prominently feature their own names. This observation might stem from viewers either mentioning the news outlet in their comments or replying directly to the respective YouTube channels. In either case, it indicates that the viewer has a desire to engage with a particular new outlet. Additionally, we observe that 'BBC' is present as a significant unigram in the CNN comments, and 'game changer' appears in both BCC and Sky News comments. The bigrams associated with BBC, CNN, and Sky News capture the topic of war, featuring phrases like 'war crime,' 'dragon teeth,' and 'abrams tank.' In contrast, the bigrams related to Fox News focus on domestic news, as evident in the phrase 'impeach biden.' It is worth noting that there is as absence of mentions of 'ukraine,' and only a single instance of the word 'russia' in the provided results. 

## Text Classification

First, we perform classification by comparing machine learning algorithms, including Random Forest, Linear Support Vector Classifier (LinearSVC), Multinomial Naive Bayes (MultinomialNB), and Logistic Regression with the 'multinomial' option.

We evaluate the performance of our predictive models using a 5-fold cross-validation process. Our initial dataset is randomly partitioned into five approximately equal-sized subsets, often referred to as 'folds,' with each fold containing a portion of the data points. Within each iteration, we divide these folds into 75% training data and 25% test data. During each iteration, the model is trained on the training set, composed of four of the folds, and subsequently assessed on the remaining fold, which serves as the testing set. 
The following boxplot illustrates the distribution of accuracy scores for each model. We observe that none of the models perform particularly strongly, with the multinomial logistic regression model achieving the highest accuracy score of 0.54. 

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/24822f57-3409-43f6-a2a9-54073a46ec94)

To gain deeper insights, we delve into the results specifically for the multinomial logistic regression model, examining its performance within each category. We find that that comments from Fox News and Sky News appear to be more straightforward to classify.

|      | Precision | Recall | F1-Score |  
|:---------:|:---------:|:------:|:--------:|
|  FoxNews  |   0.72    |  0.65  |   0.68   |  
|    CNN    |   0.58    |  0.53  |   0.55   |   
|    BBC    |   0.53    |  0.64  |   0.58   |   
|  SkyNews  |   0.62    |  0.58  |   0.60   |  

The confusion matrix provides insights into the model's performance. It indicates that CNN and BBC comments were frequently mistaken for each other, while Fox News comments were most commonly confused with CNN. Additionally, Sky News comments were often misclassified as BBC comments. These findings align with the correlations we observed among unigrams and bigrams. The results imply that comments from Fox News tend to be more distinct compared to those from the other three news outlets.

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/10261dbb-124c-4f0a-b65a-441d53abc4fd)

Lastly, we aim to improve these results by employing a Convolutional-NN with word embeddings. In the following plot, it is evident that the training loss is steadily decreasing, while the validation loss is showing an increase. When this pattern occurs, with the training loss continuing to decrease while the validation loss either increases or remains relatively unchanged, it is sign of overfitting. It signifies that the model is memorizing the training data rather than genuinely learning the underlying patterns present in the data. 

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/aaea114d-a6d9-4f08-a7fd-aa993b179d18)

This next plot reveals a pattern in which the training accuracy steadily increases, while the validation accuracy remains relatively constant. When the training accuracy is much higher than validation accuracy, it suggests the model has overfit the training data.

![image](https://github.com/lyndsayroach/Text-Analysis-Classification/assets/17256163/8ed39e0c-7417-4016-abdb-06aa619c9876)

## Conclusions

Despite our exploratory analysis and the diverse classification methods we applied, we faced a challenge in achieving classification accuracy higher than 54%. Nevertheless, our efforts yielded valuable insights into the underlying patterns within the YouTube comments. Notably, we found that comments on Fox News videos exhibited a distinctiveness compared to comments from other news outlets. Specifically, that some of the primary topics in Fox News comments were centred around domestic politics rather than the conflict itself, rendering these comments more identifiable.  





