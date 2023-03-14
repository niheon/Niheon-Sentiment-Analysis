# Niheon Sentiment Analysis of Amazon Echo Reviews

## Description
The Amazon Echo is a smart speaker that is designed to make the user's life easier by performing tasks such as playing music, setting alarms, and providing information about the weather and news. The device has become incredibly popular, and as a result, there are millions of reviews available on Amazon's website. In this project, we perform sentiment analysis on the reviews of the Amazon Echo to understand what customers think about the device.

Sentiment analysis is a technique used to determine the attitude of a writer or speaker towards a particular topic. In our case, we aim to classify reviews of the Amazon Echo as positive or negative based on the text of the review. We use machine learning algorithms to perform this classification.

## Getting Started
To get started with the project, you should have some knowledge of natural language processing, machine learning, and Python programming. You will also need to have a basic understanding of data preprocessing and data visualization techniques.

## Data Collection and Preparation
The data is in the form of a CSV file and has several columns such as the date the review was received, the text of the review, and the rating given by the customer. We preprocess the text data by removing stop words, converting all text to lowercase, and lemmatizing the words.

## Exploratory Data Analysis (EDA)
We perform exploratory data analysis on the preprocessed data to understand the distribution of ratings and the most common words used in positive and negative reviews. We use Plotly and Dash to visualize the data.

## Modeling
We use the preprocessed data to train several machine learning algorithms, including Naive Bayes and Random Forest, to classify the reviews as positive or negative. We use the Scikit-learn library to train and evaluate the models. We also use Latent Dirichlet Allocation (LDA) to identify topics in the reviews.

## Results
We evaluate the performance of the machine learning models using metrics such as accuracy, precision, and recall. We also use the LDA results to understand the topics discussed in the reviews. We present the results using various visualizations, including word clouds, scatter plots, and bar charts.

## Installing
To run the code, you need to have Python installed on your computer. Clone the repository and navigate to the project directory. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
## Usage
To run the Dash app, execute the following command:
```bash
python app.py
```
This will start a local server, and you can then access the web application by navigating to local host in your web browser.

Once the application is running, you can select a specific Amazon Echo product, time frame, and number of reviews to analyze. You can also input your own review and get a prediction of whether it is positive or negative.

## Demo
Try a live demo of the [app](https://sentiment-analysis-dashboard.herokuapp.com/).

## Built with
- Python: a popular programming language used for various applications, including data analysis and machine learning
- Flask: a micro web framework for building web applications in Python
- Plotly: a data visualization library used to create interactive charts and graphs
- Dash: a Python framework for building analytical web applications
- Pandas: a data analysis library used for data manipulation and analysis
- Scikit-learn: a machine learning library for building predictive models and performing data analysis tasks
- NLTK: a natural language processing library for text data analysis
- Spacy: an NLP library for advanced text analysis and language understanding
- Gensim: a topic modeling library for analyzing large text datasets
- Heroku: a cloud platform used for deploying and managing web applications

## License
This project is licensed under the MIT License - see the LICENSE file for details.
