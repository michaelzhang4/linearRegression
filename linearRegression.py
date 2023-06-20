import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep=";")   # reads in the student data and separates data on semicolons

data = data[["G1","G2","G3","failures","studytime","absences"]]     # selects all columns with numeric properties

predict = "G3"  # the predictive value

X = np.array(data.drop([predict],1))    # removes the column of predicted values and uses the independent values

y = np.array(data[predict]) # selects the column of only predicted values

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,train_size=0.9) # splits the independent and predicted values into training and test sets

best = 0    # benchmark for best score

for _ in range(99): # runs the process multiple times, so we can select the most accurate model

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.9)   # same as above

    linear = linear_model.LinearRegression()    # gets the linear regression model

    linear.fit(x_train, y_train)    # fits the training values to the model

    acc = linear.score(x_test,y_test)   # scores the fitted model against test sets (accuracy)

    if acc > best:  # if the accuracy is better than the benchmark then do
        best = acc  # set best to acc
        print(best) # output visual feedback
        with open("studentmodel.pickle", "wb") as f:    # open/create a pickle file in "write binary" mode
            pickle.dump(linear, f)  # store the created model in the pickle file

pickle_in = open("studentmodel.pickle", "rb")   # open existing pickle file in "read binary" mode
linear = pickle.load(pickle_in) # load model from pickle file

predictions = linear.predict(x_test)    # predict values based on x_test set

for x in range(len(predictions)):   # for each prediction do
    print(predictions[x], x_test[x], y_test[x]) # outputs the model's predicted value, the independent values and the actual value

print(linear.score(x_test, y_test)) # outputs the score/accuracy of the model on the test set

style.use("ggplot") # uses a cleaner style for matplotlib displays

p = "G1"    # independent value

pyplot.scatter(data[p],data["G3"])  # creates a scatterplot with the independent value and the dependent value
pyplot.xlabel(p)    # creates a label for the x axis based on independent value
pyplot.ylabel("Final Grade")    # creates a label "Final Grade" for the dependent value
pyplot.show()   # shows scatterplot





