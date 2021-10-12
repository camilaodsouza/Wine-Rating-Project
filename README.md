# Datarevenue Code Challenge - Wine Rating Predictor

In this project, we built a proof of concept **Wine Rating Predictor**. 

To train our machine learning model, we used both some of the features that originally came with the dataset, but also numerical features that were extracted from originally textual features. 

Using a basic linear regression model as our baseline, we achieved a 5.0 MSE in our regression problem. We were able to improve this performance by using a **XGBoost** model with default hyperparameters, which achieved a **3.55 MSE**. 

Our pipeline runs in with Docker and Luigi and can be run on any machine with docker and docker compose.
Five tasks are run in cascade: 
1. Download Data
2. Clean Data
3. Make Dataset
4. Train Model
5. Evaluate Model 

Command to build images: 
`./build-task-images.sh 0.1`
Command to run the pipeline:
`docker-compose up orchestrator` 

Download Data - Downloads the Wine Rating Dataset

Clean Data - Cleans dataset, drops irrelevant features and deals with null values. The output of the task is a 'Clean.csv' dataset in the '/data_root/interim' folder.

Make Dataset - Splits the clean dataset into 80% for the train set and 20% for test set. The output of the task ia a 'Test.csv' and a 'Train.csv' dataset in the 'data_root/partition' folder.

Train Model - Trains a XGBoost model on the train dataset. The output of the task is a 'trained_model.sav' file in the 'data_root/model' folder. This .sav file is the serialized model.

Evaluate Model - Evaluates the model, considering the MSE metric. The output of the task is a 'report.pdf file with the model metrics and most important graphs.

We believe that this result can be further improved by tuning both the hyperparameter and the VADER sentiment analysis tool to our specific case. More experiments with different types of encoders can also help us achieve higher result. 
We recommend implementing a full production solution, not only for the promising predictor metrics, but also for its potential to provide important insight to our costumer about their products and comercial partners.
