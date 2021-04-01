# Disaster Response Pipeline Project

## Project motivation:
In this project, I use disaster response data from figure8 to build a classification model to classify responses into different categories. I also build an API and front end to display these response. A custom query can be submitted as well.

During the disaster, people use twitter, facebook or other social media platformt to send messages whether it is for help or some other generic messages. If we could classify these messages into corresponding categories, respective authorities could be notified and appropriate response can be taken.

## File Description:
    - app:
    |  - templates
    | |- master.html # main html template
    | |- go.html # Display classification result
    | - run.py # Python file to run the application using flask
    
    - data
    |  - disaster_categories.csv # csv file with disaster response categories
    |  - disaster_messages.csv # Csv file with disaster responses
    |  - DisasterResponse.db # SQL processed file
    |  - process.py # python script to process disaster data
    
    - models
    | - classifier.pkl # classifier model file
    | - train_classifier.py # script to train a classifier and store the model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:8080/
