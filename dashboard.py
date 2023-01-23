import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
"""from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) """
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 
#import libraries 
import re
import string
import nltk

from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#connect to google drive
"""from google.colab import drive
drive.mount('/content/drive')"""

# Load the data
reviews_df = pd.read_csv('amazon_reviews.csv') #you should change the path 
reviews_df

# Load spacy
nlp = spacy.load('en_core_web_sm')

def clean_string(text, stem="None"):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)
    

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    
    useless_words = ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]


    if stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]

    final_string = ' '.join(text_stemmed)

    return final_string

reviews_df['verified_reviews'] = reviews_df['verified_reviews'].apply(lambda x: clean_string(x, stem='Lem')) #apply the function clean_string
reviews_df['verified_reviews'] = reviews_df['verified_reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) #remove stop words from the column.

reviews_df['variation'].value_counts()

# View the DataFrame Information
reviews_df.info()

import pandas as pd   #create the Date receiced column that will be used later in the dashbboard
reviews_df['Date received'] = pd.to_datetime(reviews_df['date'].str.strip(), infer_datetime_format=True)
reviews_df['Date received'] = pd.to_datetime(reviews_df['Date received'].dt.strftime('%m-%d-%Y'))

min_date = reviews_df["Date received"].min()
max_date = reviews_df["Date received"].max()
max_date
#these values will be used later in the dashboard

reviews_df  # format="%m/%d/%Y" #this format is required in the dash

# View DataFrame Statistical Summary
reviews_df.describe()

"""**MINI CHALLENGE #1:** 
- **Drop the 'date' column from the DataFrame** 
- **Ensure that the column has been succesfully dropped** 
"""

reviews_df = reviews_df.drop(['date'], axis=1)
reviews_df.head()

"""# Bigram Part"""

#bigrams_EDA , cleaning for the bigram extraction task
bigram_freq = lambda s: list(nltk.FreqDist(nltk.bigrams(s.split(" "))).items())
reviewss_df = reviews_df.copy()
reviewss_df['bigrams']= reviewss_df['verified_reviews'].apply(bigram_freq)
reviewss_df =reviewss_df.explode('bigrams')
reviewss_df['bigram'], reviewss_df['b'] = reviewss_df.bigrams.str

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030" "]+", re.UNICODE)
    return re.sub(emoj, '', str(data)) 

reviewss_df['bigram'] = reviewss_df.apply(lambda x: remove_emojis(x.bigram), axis=1)
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace("(", "")
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace("[!@#$]", "") 
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace(")", "")
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace("'", "")
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace("'", "")
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace(",", "_")
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace("_ ", "_")
reviewss_df['bigram'] = reviewss_df['bigram'].astype(str).str.replace(" _", "_")
del reviewss_df['b']
bigram_df =reviewss_df.groupby(['bigram','variation']).size().reset_index() 
bigram_df = bigram_df.rename(columns={0: 'value', 'variation': 'company','bigram':'ngram'})# the output is a dataframe with bigrams grouped by varitation with count.

bigram_df

bigram_df = bigram_df[bigram_df.ngram != '_']
bigram_df = bigram_df[bigram_df.ngram != '_a']
bigram_df = bigram_df[bigram_df.ngram != '_not']
bigram_df = bigram_df[bigram_df.ngram != '_and']
bigram_df = bigram_df[bigram_df.ngram != ' _not']
bigram_df = bigram_df[bigram_df.ngram != ' _and']
bigram_df = bigram_df[bigram_df.ngram != ' _are']

#cleanning 
bigram_df = bigram_df[bigram_df.ngram != 'nan']
bigram_df['ngram'] = bigram_df['ngram'].astype(str).str.replace("—", " ")
bigram_df['ngram'] = bigram_df['ngram'].astype(str).str.replace("“", " ")
bigram_df['ngram'] = bigram_df['ngram'].astype(str).str.replace("‘", " ")
bigram_df['ngram'] = bigram_df['ngram'].astype(str).str.replace("’", " ")
bigram_df['ngram'] = bigram_df['ngram'].astype(str).str.replace(" ", "")
bigram_df['ngram'] = bigram_df['ngram'].astype(str).str.replace("”", " ")
bigram_df

embed_df = bigram_df.copy()   #embed_df will be used later in the dashboard, is just a groupby dataframe 
#embed_df = embed_df.ngram.value_counts()
embed_df = embed_df.groupby('ngram').count().reset_index()
embed_df = embed_df.sort_values(
   by="company",
    ascending=False
)
embed_df = embed_df.rename(columns={'ngram': 'bigram', 'company': 'count','value':'words'}) #just remaining for the dashboard
embed_df = embed_df.iloc[:22]  #take only the 21 most frequent bigrams that will be plotted in the dashboard later
embed_df

"""# TASK #3: PERFORM DATA VISUALIZATION"""

reviews_df['variation'].value_counts()

# Check for missing values
reviews_df.isnull()

# Check for missing with a heatmap to confirm
sns.heatmap(reviews_df.isnull(), yticklabels = False)

# Plot the count plot for the ratings 
sns.countplot(x = reviews_df['rating'])

"""**MINI CHALLENGE #2:** 
- **Plot the countplot for the feedback column**
- **Roughly how many positive and negative feedback are present in the dataset?**
"""

# Plot the count plot for the feedback
sns.countplot(x = reviews_df['feedback'])

"""# TASK #4: PERFORM DATA EXPLORATION"""

# Get the length of characters for each verfied review
reviews_df['length'] = (reviews_df['verified_reviews']).apply(len)

reviews_df

# Plot the histogram for the length
reviews_df['length'].plot(bins = 100, kind = 'hist')

# Apply the describe method to get statistical summary
reviews_df.describe()

# Let's see the longest message 
reviews_df[reviews_df['length'] == 2851]

# Grab only the verified reviews column and show the first element
#reviews_df[reviews_df['length'] == 2851]['verified_reviews'].iloc[0]

"""**MINI CHALLENGE #3:**
- **View the message with the average length**
"""

# View the message with the average length
reviews_df[reviews_df['length'] == 132]['verified_reviews'].iloc[0]

"""# TASK #5: PLOT THE WORDCLOUD"""

# Obtain only the positive reviews
positive = reviews_df[reviews_df['feedback']==1]

positive

# Obtain the negative reviews only
negative = reviews_df[reviews_df['feedback']==0]

negative

# Convert to list format
sentences = positive['verified_reviews'].tolist()
len(sentences)

type(sentences)

# Join all reviews into one large string
sentences_as_one_string = ' '.join(sentences)

type(sentences_as_one_string)

from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

"""**MINI CHALLENGE #4:** 
- **Plot the wordcloud of the "negative" dataframe** 
- **What do you notice? Does the data make sense?**
"""

# Plot the wordcloud of the "negative" dataframe
sentences = negative['verified_reviews'].tolist() # Convert to list format
sentences_as_one_string = ' '.join(sentences)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

"""# TASK #6: TEXT DATA CLEANING 101"""

import string
string.punctuation

Test = '$I Love Coursera &Rhyme Guided Projects...!!!!'

Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed

# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join

import nltk # Natural Language tool kit

# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('english')
STOPWORDS = stopwords.words('english') #used in the dash plotly

Test_punc_removed_join = 'I have been enjoying these coding, programming and AI guided Projects on Rhyme and Coursera'

Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

Test_punc_removed_join_clean

"""**MINI CHALLENGE #5:** 
- **For the following text, create a pipeline to remove punctuations followed by removing stopwords and test the pipeline**
- **mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations from text..!!'**
"""

mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations from text..!!'

import string

challege = [ char for char in mini_challenge  if char not in string.punctuation ]
challenge = ''.join(challege)
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 
challenge

"""# TASK #7: PERFORM COUNT VECTORIZATION (TOKENIZATION)

![image.png](attachment:image.png)
"""

from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

print(vectorizer.get_feature_names())

print(X.toarray())

"""**MINI CHALLENGE #6:**
- **Without doing any code, perform count vectorization for the following list:**
    -  mini_challenge = ['Hello World','Hello Hello World','Hello World world world']
- **Confirm your answer with code**
"""

mini_challenge = ['Hello World','Hello Hello World','Hello World world world']
vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer.fit_transform(mini_challenge)
print(X_challenge.toarray())

"""# TASK #8: CREATE A PIPELINE TO REMOVE PUNCTUATIONS, STOPWORDS AND PERFORM COUNT VECTORIZATION"""

# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def process_text(text):
    test_punc_removed = [char for char in text if char not in string.punctuation]
    test_punc_removed = ''.join(test_punc_removed)
    test_punc_removed = [word for word in test_punc_removed.split() if word.lower() not in stopwords.words('english')]
    
    return test_punc_removed

# Let's test the newly added function
reviews_df_clean = reviews_df['verified_reviews'].apply(process_text)

# show the original review
print(reviews_df['verified_reviews'][5])

# show the cleaned up version
print(reviews_df_clean[5])

from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = process_text)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])

print(vectorizer.get_feature_names())

print(reviews_countvectorizer.toarray())

reviews_countvectorizer.shape

reviews = pd.DataFrame(reviews_countvectorizer.toarray())

X = reviews

X

y = reviews_df['feedback']
y

"""**MINI CHALLENGE #7:**
- **What is the shape of X and Y**
"""

X.shape, y.shape

"""# TASK #9: TRAIN AND TEST NAIVE BAYES CLASSIFIER MODEL"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

"""![image.png](attachment:image.png)"""

from sklearn.metrics import classification_report, confusion_matrix

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict_test))

"""**MINI CHALLENGE #8:**
- **Train a logistic Regression classifier and assess its performance**
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))

"""# EXCELLENT JOB! YOU SHOULD BE PROUD OF YOUR NEWLY ACQUIRED SKILLS"""



"""# MINI CHALLENGE SOLUTIONS

**MINI CHALLENGE #2 SOLUTION:** 
- **Plot the countplot for the feedback column**
- **Roughly how many positive and negative feedback are present in the dataset?**
"""

# Plot the countplot for feedback
# Positive ~2800
# Negative ~250
sns.countplot(x = reviews_df['feedback'])

"""**MINI CHALLENGE #3 SOLUTION:**
- **View the message with the average length**
"""

# Let's see the message with mean length 
reviews_df[reviews_df['length'] == 132]['verified_reviews'].iloc[0]

"""**MINI CHALLENGE #4 SOLUTION:** 
- **Plot the wordcloud of the "negative" dataframe** 
- **What do you notice? Does the data make sense?**
"""

sentences = negative['verified_reviews'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize = (20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

"""**MINI CHALLENGE #5 SOLUTION:** 
- **For the following text, create a pipeline to remove punctuations followed by removing stopwords and test the pipeline**
- **mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations from text..!!'**
"""

mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations from text..!!'
challege = [ char for char in mini_challenge  if char not in string.punctuation ]
challenge = ''.join(challege)
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 
challenge

"""**MINI CHALLENGE #6 SOLUTION:**
- **Without doing any code, perform count vectorization for the following list:**
    -  mini_challenge = ['Hello World','Hello Hello World','Hello World world world']
- **Confirm your answer with code**
"""

mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

mini_challenge = ['Hello World', 'Hello Hello Hello World world', 'Hello Hello World world world World']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())

"""**MINI CHALLENGE #7 SOLUTION:**
- **What is the shape of X and Y**
"""

X.shape

y.shape

"""**MINI CHALLENGE #8 SOLUTION:**
- **Train a logistic Regression classifier and assess its performance**
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))

"""# Excellent Job!

# LDA PART
"""

import spacy  # for our NLP processing
import nltk  # to use the stopwords library
import string  # for a list of all punctuation
from nltk.corpus import stopwords  # for a list of stopwords
import gensim
from sklearn.manifold import TSNE
import pathlib
import pandas as pd
from wordcloud import STOPWORDS
import re
import json

# Now we can load and use spacy to analyse our reviews
nlp = spacy.load("en_core_web_sm")


def format_topics_sentences(ldamodel, corpus, texts, dates):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )
            else:
                break
    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents, pd.Series(dates)], axis=1)
    return sent_topics_df


def lda_analysis(df, stop_words):
    
    def cleanup_text(doc):
        doc = nlp(doc, disable=["parser", "ner"])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
        tokens = [
            tok for tok in tokens if tok not in stop_words and tok not in punctuations
        ]
        return tokens

    # Clean up and take only rows where we have text
    df = df[pd.notnull(df["verified_reviews"])]
    docs = list(df["verified_reviews"].values)

    punctuations = string.punctuation

    processed_docs = list(map(cleanup_text, docs))
    print("len(processed_docs)", len(processed_docs))
    if len(processed_docs) < 11:
        print("INSUFFICIENT DOCS TO RUN LINEAR DISCRIMINANT ANALYSIS")
        return (None, None, None, None)

    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    print("len(bow_corpus)", len(bow_corpus))
    print("dictionary", len(list(dictionary.keys())))
    if len(list(dictionary.keys())) < 1:
        print("INSUFFICIENT DICTS TO RUN LINEAR DISCRIMINANT ANALYSIS")
        return (None, None, None, None)

    lda_model = gensim.models.LdaModel(
        bow_corpus, num_topics=5, id2word=dictionary, passes=10
    )

    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=lda_model,
        corpus=bow_corpus,
        texts=docs,
        dates=list(df["Date received"].values),
    )
    print("len(df_topic_sents_keywords)", len(df_topic_sents_keywords))
    print("df_topic_sents_keywords.head()", df_topic_sents_keywords.head())
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = [
        "Document_No",
        "Dominant_Topic",
        "Topic_Perc_Contrib",
        "Keywords",
        "Text",
        "Date",
    ]

    topic_num, tsne_lda = tsne_analysis(lda_model, bow_corpus)

    return (tsne_lda, lda_model, topic_num, df_dominant_topic)


def tsne_analysis(ldamodel, corpus):
    topic_weights = []
    for i, row_list in enumerate(ldamodel[corpus]):
        topic_weights.append([w for i, w in row_list])

    # Array of topic weights
    df_topics = pd.DataFrame(topic_weights).fillna(0).values



    # Dominant topic number in each doc
    topic_nums = np.argmax(df_topics, axis=1)

    # tSNE Dimension Reduction
    try:
        tsne_model = TSNE(
            n_components=2, verbose=1, random_state=0, angle=0.99, init="pca"
        )
        tsne_lda = tsne_model.fit_transform(df_topics)
    except:
        print("TSNE_ANALYSIS WENT WRONG, PLEASE RE-CHECK YOUR DATASET")
        return (topic_nums, None)

    return (topic_nums, tsne_lda)

GLOBAL_DF = reviews_df.copy()

ADDITIONAL_STOPWORDS = [
    "XXXX",
    "XX",
    "xx",
    "xxxx",
    "n't"
]
for stopword in ADDITIONAL_STOPWORDS:
    STOPWORDS.add(stopword)

def add_stopwords(selected_bank):

    selected_bank_words = re.findall(r"[\w']+", selected_bank)
    for word in selected_bank_words:
        STOPWORDS.add(word.lower())

    print("Added %s stopwords:" % selected_bank)
    for word in selected_bank_words:
        print("\t", word)
    return STOPWORDS

def precompute_all_lda():
    """ QD function for precomputing all necessary LDA results
     to allow much faster load times when the app runs. """

    failed_banks = []
    counter = 0
    bank_names = GLOBAL_DF["variation"].value_counts().keys().tolist()
    results = {}

    for bank in bank_names:
        try:
            print("crunching LDA for: ", bank)
            add_stopwords(bank)
            bank_df = GLOBAL_DF[GLOBAL_DF["variation"] == bank]
            tsne_lda, lda_model, topic_num, df_dominant_topic = lda_analysis(
                bank_df, list(STOPWORDS)
            )

            topic_top3words = [
                (i, topic)
                for i, topics in lda_model.show_topics(formatted=False)
                for j, (topic, wt) in enumerate(topics)
                if j < 3
            ]

            df_top3words_stacked = pd.DataFrame(
                topic_top3words, columns=["topic_id", "words"]
            )
            df_top3words = df_top3words_stacked.groupby("topic_id").agg(", \n".join)
            df_top3words.reset_index(level=0, inplace=True)

            # print(len(tsne_lda))
            # print(len(df_dominant_topic))
            tsne_df = pd.DataFrame(
                {
                    "tsne_x": tsne_lda[:, 0],
                    "tsne_y": tsne_lda[:, 1],
                    "topic_num": topic_num,
                    "doc_num": df_dominant_topic["Document_No"],
                }
            )

            topic_top3words = [
                (i, topic)
                for i, topics in lda_model.show_topics(formatted=False)
                for j, (topic, wt) in enumerate(topics)
                if j < 3
            ]

            df_top3words_stacked = pd.DataFrame(
                topic_top3words, columns=["topic_id", "words"]
            )
            df_top3words = df_top3words_stacked.groupby("topic_id").agg(", \n".join)
            df_top3words.reset_index(level=0, inplace=True)

            results[str(bank)] = {
                "df_top3words": df_top3words.to_json(),
                "tsne_df": tsne_df.to_json(),
                "df_dominant_topic": df_dominant_topic.to_json(),
            }

            counter += 1
        except:
            print("SOMETHING WENT HORRIBLY WRONG WITH : ", bank)
            failed_banks.append(bank)

    with open("precomputed.json", "w+") as res_file:
        json.dump(results, res_file)

    print("DONE")
    print("did %d variations" % counter)
    print("failed %d:" % len(failed_banks))
    for fail in failed_banks:
        print(fail)

precompute_all_lda()

"""# EMBEDDING VECTORS PART"""

vects_df = embed_df.copy()

#words converted to sentences by tfidf weights.
#Step 1. Prepare data
#step 2. Have bogus word2vec (of the size of our vocab)
#Step 3. Calculate a column containing word2vec for sentences

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(vects_df.bigram).todense()
vocab = tfidf.vocabulary_
vocab

word2vec = np.random.randn(len(vocab),300)

sent2vec_matrix = np.dot(tfidf_matrix, word2vec) # word2vec here contains vectors in the same order as in vocab
vects_df["ngram2vec"] = sent2vec_matrix.tolist()

vects_df

vects_df = vects_df.ngram2vec.apply(pd.Series) #split horizontally

vects_df

"""# DASH PART"""

#!pip install dash pip install emoji
#!pip install dash_bootstrap_components
#!pip install jupyter-dash

#since we are using google colab/jupyter we should use JupyterDash
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np 
import plotly.graph_objs as go
import emoji
import pandas as pd
import plotly
from plotly.offline import iplot
from plotly.offline import init_notebook_mode, iplot
from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
import cufflinks as cf
init_notebook_mode(connected=True)
import plotly.express as px
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
from wordcloud import WordCloud, STOPWORDS
from sklearn.manifold import TSNE
import pathlib
import json
import pathlib
import re
import json
from datetime import datetime
import flask

#app = dash.Dash(__name__)   #,server= server , routes_pathname_prefix='/dash/' 
#app = JupyterDash(__name__)  #since we are using google colab/jupyter we should use JupyterDash
#__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__)
server = app.server  # for Heroku deployment

DATA_PATH = "/content/drive/MyDrive/Dash_plotly_home"
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
FILENAME_PRECOMPUTED = "/precomputed.json"
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
with open("precomputed.json") as precomputed_file:
    PRECOMPUTED_LDA = json.load(precomputed_file)

GLOBAL_DF1 = reviews_df
"""
We are casting the whole column to datetime to make life easier in the rest of the code.
It isn't a terribly expensive operation so for the sake of tidyness we went this way.
"""
GLOBAL_DF["Date received"] = pd.to_datetime(
    GLOBAL_DF["Date received"], format="%m/%d/%Y"
)

"""
In order to make the graphs more useful we decided to prevent some words from being included
"""
ADDITIONAL_STOPWORDS = [
    "XXXX",
    "XX",
    "xx",
    "xxxx",
    "n't",
]
for stopword in ADDITIONAL_STOPWORDS:
    STOPWORDS.add(stopword)

# Commented out IPython magic to ensure Python compatibility.
def sample_data(dataframe, float_percent):
    """
    Returns a subset of the provided dataframe.
    The sampling is evenly distributed and reproducible
    """
    print("making a local_df data sample with float_percent: %s" % (float_percent))
    return dataframe.sample(frac=float_percent, random_state=1)


def get_complaint_count_by_company(dataframe):
    """ Helper function to get review counts for unique variations """
    company_counts = dataframe["variation"].value_counts()
    # we filter out all banks with less than 11 reviews for now
    company_counts = company_counts[company_counts > 10]
    values = company_counts.keys().tolist()
    counts = company_counts.tolist()
    return values, counts


def calculate_bank_sample_data(dataframe, sample_size, time_values):
    """ TODO """
    print(
        "making reviews_sample_data with sample_size count: %s and time_values: %s"
#         % (sample_size, time_values)
    )
    if time_values is not None:
        min_date = time_values[0]
        max_date = time_values[1]
        dataframe = dataframe[
            (dataframe["Date received"] >= min_date)
            & (dataframe["Date received"] <= max_date)
        ]
    company_counts = dataframe["variation"].value_counts()
    company_counts_sample = company_counts[:sample_size]
    values_sample = company_counts_sample.keys().tolist()
    counts_sample = company_counts_sample.tolist()

    return values_sample, counts_sample


def make_local_df(selected_bank, time_values, n_selection):
    """ TODO """
    print("redrawing dataset-wordcloud...")
    n_float = float(n_selection / 100)
    print("got time window:", str(time_values))
    print("got n_selection:", str(n_selection), str(n_float))
    # sample the dataset according to the slider
    local_df = sample_data(reviews_df, n_float)
    if time_values is not None:
        time_values = time_slider_to_date(time_values)
        local_df = local_df[
            (local_df["Date received"] >= time_values[0])
            & (local_df["Date received"] <= time_values[1])
        ]
    if selected_bank:
        local_df = local_df[local_df["variation"] == selected_bank]
        
    return local_df


def make_marks_time_slider(mini, maxi):
    """
    A helper function to generate a dictionary that should look something like:
    {1420066800: '2015', 1427839200: 'Q2', 1435701600: 'Q3', 1443650400: 'Q4',
    1451602800: '2016', 1459461600: 'Q2', 1467324000: 'Q3', 1475272800: 'Q4',
     1483225200: '2017', 1490997600: 'Q2', 1498860000: 'Q3', 1506808800: 'Q4'}
    """
    step = relativedelta.relativedelta(months=+1)
    start = datetime(year=mini.year, month=1, day=1)
    end = datetime(year=maxi.year, month=maxi.month, day=30)
    ret = {}

    current = start
    while current <= end:
        current_str = int(current.timestamp())
        if current.month == 1:
            ret[current_str] = {
                "label": str(current.year),
                "style": {"font-weight": "bold"},
            }
        elif current.month == 4:
            ret[current_str] = {
                "label": "Q2",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        elif current.month == 7:
            ret[current_str] = {
                "label": "Q3",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        elif current.month == 10:
            ret[current_str] = {
                "label": "Q4",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        else:
            pass
        current += step
    # print(ret)
    return ret


def time_slider_to_date(time_values):
    """ TODO """
    min_date = datetime.fromtimestamp(time_values[0]).strftime("%c")
    max_date = datetime.fromtimestamp(time_values[1]).strftime("%c")
    print("Converted time_values: ")
    print("\tmin_date:", time_values[0], "to: ", min_date)
    print("\tmax_date:", time_values[1], "to: ", max_date)
    return [min_date, max_date]


def make_options_bank_drop(values):
    """
    Helper function to generate the data format the dropdown dash component wants
    """
    ret = []
    for value in values:
        ret.append({"label": value, "value": value})
    return ret

import matplotlib.colors as mcolors
def populate_lda_scatter(tsne_df, df_top3words, df_dominant_topic):
    """Calculates LDA and returns figure data you can jam into a dcc.Graph()"""
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

    # for each topic we create a separate trace
    traces = []
    for topic_id in df_top3words["topic_id"]:
        tsne_df_f = tsne_df[tsne_df.topic_num == topic_id]
        cluster_name = ", ".join(
            df_top3words[df_top3words["topic_id"] == topic_id]["words"].to_list()
        )
        trace = go.Scatter(
            name=cluster_name,
            x=tsne_df_f["tsne_x"],
            y=tsne_df_f["tsne_y"],
            mode="markers",
            hovertext=tsne_df_f["doc_num"],
            marker=dict(
                size=6,
                color=mycolors[tsne_df_f["topic_num"]],  # set color equal to a variable
                colorscale="Viridis",
                showscale=False,
            ),
        )
        traces.append(trace)

    layout = go.Layout({"title": "Topic analysis using LDA"})

    return {"data": traces, "layout": layout}


def plotly_wordcloud(data_frame):
    """A wonderful function that returns figure data for three equally
    wonderful plots: wordcloud, frequency histogram and treemap"""
    complaints_text = list(data_frame["verified_reviews"].dropna().values)

    if len(complaints_text) < 1:
        return {}, {}, {}

    # join all documents in corpus
    text = " ".join(list(complaints_text))

    word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=80, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:15]
    word_list_top.reverse()
    freq_list_top = freq_list[:15]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }
    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    return wordcloud_figure_data, frequency_figure_data, treemap_figure


"""
#  Page layout and contents
In an effort to clean up the code a bit, we decided to break it apart into
sections. For instance: LEFT_COLUMN is the input controls you see in that gray
box on the top left. The body variable is the overall structure which most other
sections go into. This just makes it ever so slightly easier to find the right
spot to add to or change without having to count too many brackets.
"""

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("NLP DASHBOARD for reviews classification", className="ml-2")
                    ),
                ],
                align="center",
                
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

LEFT_COLUMN = dbc.Col(html.Div(
    [
        html.H4(children="Select a variation & dataset size", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select percentage of dataset", className="lead"),
        html.P(
            "(Lower is faster. Higher is more precise)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Slider(
            id="n-selection-slider",
            min=1,
            max=100,
            step=1,
            marks={
                0: "0%",
                10: "",
                20: "20%",
                30: "",
                40: "40%",
                50: "",
                60: "60%",
                70: "",
                80: "80%",
                90: "",
                100: "100%",
            },
            value=20,
        ),
        html.Label("Select a variation", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="bank-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider"), style={"marginBottom": 50}),
        html.P(
            "(You can define the time frame down to month granularity)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
    ])
)

LDA_PLOT = dcc.Loading(
    id="loading-lda-plot", children=[dcc.Graph(id="tsne-lda")], type="default"
)
LDA_TABLE = html.Div(
    id="lda-table-block",
    children=[
        dcc.Loading(
            id="loading-lda-table",
            children=[
                dash_table.DataTable(
                    id="lda-table",
                    style_cell_conditional=[
                        {
                            "if": {"column_id": "Text"},
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "min-width": "50%",
                        }
                    ],
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(243, 246, 251)",
                        }
                    ],
                    style_cell={
                        "padding": "16px",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "max-width": "0",
                    },
                    style_header={"backgroundColor": "white", "fontWeight": "bold"},
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    filter_action="native",
                    page_action="native",
                    page_current=0,
                    page_size=5,
                    columns=[],
                    data=[],
                )
            ],
            type="default",
        )
    ],
    style={"display": "none"},
)

LDA_PLOTS = [
    dbc.CardHeader(html.H5("Topic modelling using LDA")),
    dbc.Alert(
        "Not enough data to render LDA plots, please adjust the filters",
        id="no-data-alert-lda",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            html.P(
                "Click on a review point in the scatter to explore that specific reveiw",
                className="mb-0",
            ),
            html.P(
                "(not affected by sample size or time frame selection)",
                style={"fontSize": 10, "font-weight": "lighter"},
            ),
            LDA_PLOT,
            html.Hr(),
            LDA_TABLE,
        ]
    ),
]
WORDCLOUD_PLOTS = [
    dbc.CardHeader(html.H5("Most frequently used words in reviews")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="bank-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(id="bank-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        md=8,
                    ),
                ]
            )
        ]
    ),
]

TOP_BANKS_PLOT = [
    dbc.CardHeader(html.H5("Top variations by number of verified reviews")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-banks-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-bank",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="bank-sample"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BIGRAM_PLOT = [
    dbc.CardHeader(html.H5("Top bigrams found in the database")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-scatter",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P(["Choose a t-SNE perplexity value:"]), md=6),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-perplex-dropdown",
                                        options=[
                                            {"label": str(i), "value": i}
                                            for i in range(3, 7)
                                        ],
                                        value=3,
                                    )
                                ],
                                md=3,
                            ),
                        ]
                    ),
                    dcc.Graph(id="bigrams-scatter"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BIGRAM_COMPS = [
    dbc.CardHeader(html.H5("Comparison of bigrams for two variations")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-comps",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams_comp",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P("Choose two variations to compare:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-comp_1",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in bigram_df.company.unique()
                                        ],
                                        value="Charcoal Fabric",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-comp_2",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in bigram_df.company.unique()
                                        ],
                                        value="Heather Gray Fabric",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="bigrams-comps"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_COMPS)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_PLOT)),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(TOP_BANKS_PLOT), md=8),
            ],
            style={"marginTop": 30},
        ),
        dbc.Card(WORDCLOUD_PLOTS),
        dbc.Row([dbc.Col([dbc.Card(LDA_PLOTS)])], style={"marginTop": 50}),
    ],
    className="mt-12",
)

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""


@app.callback(
    Output("bigrams-scatter", "figure"), [Input("bigrams-perplex-dropdown", "value")],
)
def populate_bigram_scatter(perplexity):
    X_embedded = TSNE(n_components=2, perplexity=perplexity).fit_transform(vects_df)

    embed_df["tsne_1"] = X_embedded[:, 0]
    embed_df["tsne_2"] = X_embedded[:, 1]
    fig = px.scatter(
        embed_df,
        x="tsne_1",
        y="tsne_2",
        hover_name="bigram",
        text="bigram",
        size="count",
        color="words",
        size_max=45,
        template="plotly_white",
        title="Bigram similarity and frequency",
        labels={"words": "Count<BR>(words)"},
        color_continuous_scale=px.colors.sequential.Sunsetdark,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="Gray")))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


@app.callback(
    Output("bigrams-comps", "figure"),
    [Input("bigrams-comp_1", "value"), Input("bigrams-comp_2", "value")],
)
def comp_bigram_comparisons(comp_first, comp_second):
    comp_list = [comp_first, comp_second]
    temp_df = bigram_df[bigram_df.company.isin(comp_list)]
    temp_df.loc[temp_df.company == comp_list[-1], "value"] = -temp_df[
        temp_df.company == comp_list[-1]
    ].value.values

    fig = px.bar(
        temp_df,
        title="Comparison: " + comp_first + " | " + comp_second,
        x="ngram",
        y="value",
        color="company",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"company": "variat°:", "ngram": "N-Gram"},
        hover_data="",
    )
    fig.update_layout(legend=dict(x=0.1, y=1.1), legend_orientation="h")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig


@app.callback(
    [
        Output("time-window-slider", "marks"),
        Output("time-window-slider", "min"),
        Output("time-window-slider", "max"),
        Output("time-window-slider", "step"),
        Output("time-window-slider", "value"),
    ],
    [Input("n-selection-slider", "value")],
)
def populate_time_slider(value):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """
    value += 0
    min_date = reviews_df["Date received"].min()
    max_date = reviews_df["Date received"].max()

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )


@app.callback(
    Output("bank-drop", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value")],
)
def populate_bank_dropdown(time_values, n_value):
    """ TODO """
    print("variation-drop: TODO USE THE TIME VALUES AND N-SLIDER TO LIMIT THE DATASET")
    if time_values is not None:
        pass
    n_value += 1
    bank_names, counts = get_complaint_count_by_company(reviews_df)
    counts.append(1)
    return make_options_bank_drop(bank_names)


@app.callback(
    [Output("bank-sample", "figure"), Output("no-data-alert-bank", "style")],
    [Input("n-selection-slider", "value"), Input("time-window-slider", "value")],
)
def update_bank_sample_plot(n_value, time_values):
    """ TODO """
    print("redrawing variation-sample...")
    print("\tn is:", n_value)
    print("\ttime_values is:", time_values)
    if time_values is None:
        return [{}, {"display": "block"}]
    n_float = float(n_value / 100)
    bank_sample_count = 10
    local_df = sample_data(reviews_df, n_float)
    min_date, max_date = time_slider_to_date(time_values)
    values_sample, counts_sample = calculate_bank_sample_data(
        local_df, bank_sample_count, [min_date, max_date]
    )
    data = [
        {
            "x": values_sample,
            "y": counts_sample,
            "text": values_sample,
            "textposition": "auto",
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": False,
        "margin": dict(t=10, b=10, l=40, r=0, pad=4),
        "xaxis": {"showticklabels": False},
    }
    print("redrawing variation-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]


@app.callback(
    [
        Output("lda-table", "data"),
        Output("lda-table", "columns"),
        Output("tsne-lda", "figure"),
        Output("no-data-alert-lda", "style"),
    ],
    [Input("bank-drop", "value"), Input("time-window-slider", "value")],
)
def update_lda_table(selected_bank, time_values):
    """ Update LDA table and scatter plot based on precomputed data """

    if selected_bank in PRECOMPUTED_LDA:
        df_dominant_topic = pd.read_json(
            PRECOMPUTED_LDA[selected_bank]["df_dominant_topic"]
        )
        tsne_df = pd.read_json(PRECOMPUTED_LDA[selected_bank]["tsne_df"])
        df_top3words = pd.read_json(PRECOMPUTED_LDA[selected_bank]["df_top3words"])
    else:
        return [[], [], {}, {}]

    lda_scatter_figure = populate_lda_scatter(tsne_df, df_top3words, df_dominant_topic)

    columns = [{"name": i, "id": i} for i in df_dominant_topic.columns]
    data = df_dominant_topic.to_dict("records")

    return (data, columns, lda_scatter_figure, {"display": "none"})


@app.callback(
    [
        Output("bank-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("bank-treemap", "figure"),
        Output("no-data-alert", "style"),
    ],
    [
        Input("bank-drop", "value"),
        Input("time-window-slider", "value"),
        Input("n-selection-slider", "value"),
    ],
)
def update_wordcloud_plot(value_drop, time_values, n_selection):
    """ Callback to rerender wordcloud plot """
    local_df = make_local_df(value_drop, time_values, n_selection)
    wordcloud, frequency_figure, treemap = plotly_wordcloud(local_df)
    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
        alert_style = {"display": "block"}
    print("redrawing variation wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style)


@app.callback(
    [Output("lda-table", "filter_query"), Output("lda-table-block", "style")],
    [Input("tsne-lda", "clickData")],
    [State("lda-table", "filter_query")],
)
def filter_table_on_scatter_click(tsne_click, current_filter):
    """ TODO """
    if tsne_click is not None:
        selected_complaint = tsne_click["points"][0]["hovertext"]
        if current_filter != "":
            filter_query = (
                "({Document_No} eq "
                + str(selected_complaint)
                + ") || ("
                + current_filter
                + ")"
            )
        else:
            filter_query = "{Document_No} eq " + str(selected_complaint)
        print("current_filter", current_filter)
        return (filter_query, {"display": "block"})
    return ["", {"display": "none"}]


@app.callback(Output("bank-drop", "value"), [Input("bank-sample", "clickData")])
def update_bank_drop_on_click(value):
    """ TODO """
    if value is not None:
        selected_bank = value["points"][0]["x"]
        return selected_bank
    return "Charcoal Fabric"

if __name__ == "__main__":
    #app.run_server(mode= 'inline')
    app.run_server(debug=True ,use_reloader=False)
