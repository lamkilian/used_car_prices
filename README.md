# Project Used Car Prices
This is the repository for the projet in our Data Science course (LTAT.02.002). Our project is about predicting the prices of used cars.

## Members
- Gerrit Nocker
- Kilian Lamprecht

## Data
The data of our project comes from this [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e9/overview). The competition is about predicting the price of a used car based on different features like manefacturer, milage and so on.

The data is split into two files, `train.csv` and `test.csv`, where the latter is missing the price column which is to be predicted.

## Goals
Our goal is to get as high on the leaderboard as possible. This means that we want to minimize the MSE for ourpredictions as this is the metric used by the competition. We also want to use this project to try out different possible approaches to feature engineering and model selection.

## Contents of the repository
Our data analysis and feature engineering is in the [`data_analysis`](data_analysis.ipynb)-file. It describes the pipeline we used. The same pipeline was used to transform the test data, we just added that no data points were dropped and a reordering of the features to fit the order of the training data.

[`gradient_boost.py`](gradient_boost.py) holds the pipeline used for training and saving our model. We used xgboost.

[`submission.py`] loads the model and uses it to create our [predictions](submission.csv), which we finally manually submitted to Kaggle.

The [data](data) folder contains the original data from the competition. We could not store the transformed data, as it was too large.