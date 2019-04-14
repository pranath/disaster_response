# Disaster Response Pipeline Project
In this project I will analyze disaster messages to build a machine learning model that classifies the messages using a web application so they can be sent to the relevant aid agencies.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/

## Important Files

- data/process_data.py: The ETL pipeline used to process data in preparation for model building.
- models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle (pickle is not uploaded to the repo due to size constraints.).
- app/templates/*.html: HTML templates for the web app.
- app/run.py: Start the Python server for the web app and prepare visualizations.
- notebooks/*: Juypter notebooks used to develop ETL, NLP & ML pipelines

## Screenshots

### Home page - stats view 1

![Text](../master/screenshots/response_app1.png)

### Home page - stats view 2

![Text](../master/screenshots/response_app2.png)

### Categorization results page

![Text](../master/screenshots/response_app3.png)
