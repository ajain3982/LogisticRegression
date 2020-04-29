# LogisticRegression
This repository creates a logreg model on your custom data from scratch.

## Table Of Contents
1. [Motivation](#motivation)
2. [Required Libraries](#required-libraries)
3. [Run](#run)
4. [Output](#output)

## Motivation
This project is used to train a logistic regression model from scratch. I always believe learning through code is the best method to learn Machine Learning. Therefore, to all the beginners or even advanced users, checkout the code and see how gradient descent actually works in a logistic function.

## Required Libraries
1. Python
2. Numpy
3. Pandas
4. scikit-learn

## Run
`python run.py --train iris.csv --lr 0.01 --epochs 300 `

You can use your dataset as well. Make sure the target column is names as **labels** and there are no ID columns. I am using the first column as index column, so adjust your file accordingly. A sample data is also provided.

To checkout other configurable options, run - 

`python run.py --help`

## Output
This program outputs classification report and predictions over your test dataset.
  
