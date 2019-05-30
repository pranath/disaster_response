# Disaster Response Pipeline Project
In this project I will analyze disaster messages to build a machine learning model that classifies the messages using a web application so they can be sent to the relevant aid agencies.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves a classifier 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

2. The app folder contains a web app ready to deploy to the Heroku platform, see an example here: https://disaster-response-pranath.herokuapp.com

## Important Files

- data/process_data.py: The ETL pipeline used to process data in preparation for model building.
- models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle file.
- app/templates/*.html: HTML templates for the web app.
- app/run.py: Start the Python server for the web app and prepare visualizations.
- notebooks/*: Juypter notebooks used to develop ETL, NLP & ML pipelines

## Presentations
This work was presented at the Data Science meetup 'Brighton Data Forum' on 29th May 2019

https://www.meetup.com/Brighton-Data-Forum/events/mgggbqyzhbfc/

The presentation slides for this can be found here:

https://docs.google.com/presentation/d/164oDiuEZFR35X9QEJN1R-rOb4sp5r32DxR4Xm6z-Qgs/edit#slide=id.p

## Screenshots

### Home page - stats view 1

![Text](../master/screenshots/response_app1.png)

### Home page - stats view 2

![Text](../master/screenshots/response_app2.png)

### Categorization results page

![Text](../master/screenshots/response_app3.png)
