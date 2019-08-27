# Disaster Response Pipeline Project

##  Web app where an emergency worker can input a new message and get classification results in several categories

## General Information
With the help natural language processing, machine learning and data engineering skills I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster message.
The web app will also display visualizations of the data.


## Specifications
The project includes the following files:

1. ETL Pipeline
Python script, `process_data.py`, cleaning pipeline that:

Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
Python script, `train_classifier.py`, machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
