# Disaster Response Pipeline Project

##  web app where an emergency worker can input a new message and get classification results in several categories

## Link to Webapp [https://emerging-markets-land-use.herokuapp.com](https://emerging-markets-land-use.herokuapp.com)

## General Information
With the help natural language processing, machine learning and data engineering skills I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster message.
The web app will also display visualizations of the data

The web app was delpoyed on heroku.

## Prerequisites

To install the flask app, you need:
- python3
- python packages in the requirements.txt file
 
 Install the packages with
``` 
 pip install -r requirements.txt

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
