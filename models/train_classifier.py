# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import GridSearchCV

import re
import sys


def load_data(database_filepath):
    """ Load data from database as dataframe
    
    Input: filepath to database
    Output: X (messages),
            y (categories),
            category_names (list of the categories
    
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_messages", engine)
    categories = df.columns[-36:]
    X = df['message'].values
    Y = df[categories]
    category_names = list(df.columns[-36:])
    
    return X, Y, category_names

def tokenize(text):
    
    """process text, tokenize, remove stopwords and reduces words to their root form
    
    Input: text to tokenize
    
    Output: cleaned tokens
    """
    
    #normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize text
    words = nltk.word_tokenize(text)
    
    #Remove stopwords
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    #  # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        #lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
   
    return clean_tokens


def build_model():
    """ function for building the model respect. the pipeline
    
    no Input
    
    Output: model
    """
    
    # setting up the machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])    
    
    parameters = { 
               'clf__estimator__n_estimators': [50,100],
               'clf__estimator__bootstrap': [True, False]
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)
 
    
    model = cv
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
        
    """ Report the overall accuracy of the model and the f1 score, precision and recall for each output category of the dataset
    
    Input: y_test (acutal labels)
           y_pred (predicted labels)
    
    Output:
    
    """
    #predict the labels for the test data
    Y_pred = model.predict(X_test)
    
    #accuracy of the model
    accuracy = (Y_pred == Y_test).mean().mean()
    
    # iterating through the columns and calling sklearn's classification_report on each
    Y_pred_pd = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    print("Overall Accuracy of the model: {}%\n".format(round((accuracy*100),2)))
    for column in Y_test.columns:
        print("Classification Report ()")
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_pd[column]))

#source_ https://github.com/matteobonanomi/dsnd-disaster-response/blob/master/models/ML%20Pipeline%20Preparation.ipynb
def save_model(model, model_filepath):
    """ function that stores the classifier into a pickle file to the specified model file path
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()