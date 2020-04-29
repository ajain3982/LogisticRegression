import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from logreg import LogisticRegression
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run logistic regression.')
    parser.add_argument('--train',required=True,help='path of the train file.')
    parser.add_argument('--test',help='path of the test file.',default=None)
    parser.add_argument('--lr',help='Value of learning rate.',default=0.01)
    parser.add_argument('--epochs',help='Number of epochs.',default=20)
    parser.add_argument('--init',help='Initialiser for weights. Out of xavier or he_normal.',default=None)
    parser.add_argument('--verbose',help='Print loss.',default=1)
    parser.add_argument('--output',help='Output directory',default='.')

    args = parser.parse_args()

    epochs = int(args.epochs)
    lr = float(args.lr)
    init = args.init
    verbose = int(args.verbose)
    train_file = None
    val_file = None
    test_file = None

    print(lr)
    if args.train:
        train_file = args.train
        train = pd.read_csv(train_file,index_col=0)

    if args.test:
        test_file = args.test
        test = pd.read_csv(test_file)

    if test_file is None:
        print("Splitting train to accomodate for test set.")
        train,test = train_test_split(train,test_size=0.2)

    train_Y = train['labels'].values
    train_X = train.drop(['labels'],axis=1).values

    test_Y = test['labels'].values
    test_X = test.drop(['labels'],axis=1).values

    print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)
    logreg = LogisticRegression(learning_rate=lr,epochs=epochs,initialiser=init,verbose=verbose)
    logreg.fit(train_X,train_Y)
    predictions = logreg.predict(test_X)

    if args.output == ".":
        args.output = os.getcwd()
    with open(args.output + "/classification_report.txt",'w') as f:
        f.write(str(classification_report(test_Y,predictions)))

    test['predictions'] = predictions
    test.to_csv(args.output + "/predictions.csv")
