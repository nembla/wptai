import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import os.path


#############################################################################
######################### Capacitor Prediction ##############################
#############################################################################
data = pd.read_csv("data.csv") #read in data from csv file, comma delimited
data = data[["input_power", "rout", "smith_imp_real", "smith_imp_imag", "c"]] #shows the column heading names
predict = "c" #what variable we want to predict

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
#test_size is in percentages, needs at least two points but three is better. need more input data, so this can be reduced to 0.1
#Ideally around 30 points of data would be necessary for 95% accuracy, change test size to 10%, equal accuracy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)

#Checks for existing linear model file
if os.path.isfile("matchingmodelc.pickle"):
    #Opens trained model from file
    pickle_in = open("matchingmodelc.pickle", "rb")
    linear = pickle.load(pickle_in)
    print("\nCapacitor Model")
    print('Model loaded from file')

#Generates a model and writes to a file if no file exists prior
else:
    #Code used for model training, 1k iterations
    best = 0
    for _ in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)
        linear = linear_model.LinearRegression()

        #How accurate the model is
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print(best, " ", acc)

        if acc > best:
            best = acc
            #Saves the model to a file
            with open("matchingmodelc.pickle", "wb") as f:
                pickle.dump(linear, f)
    print("\nCapacitor Model")

#print(best)
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


#############################################################################
######################### Inductor Prediction ###############################
#############################################################################
data = pd.read_csv("data.csv") #read in data from csv file, comma delimited
data = data[["input_power", "rout", "smith_imp_real", "smith_imp_imag", "l"]] #shows the column heading names
predict = "l" #what variable we want to predict

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
#test_size is in percentages, needs at least two points but three is better. need more input data, so this can be reduced to 0.1
#Ideally around 30 points of data would be necessary for 95% accuracy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)

#Checks for existing linear model file
if os.path.isfile("matchingmodell.pickle"):
    #Opens trained model from file
    pickle_in = open("matchingmodell.pickle", "rb")
    linear = pickle.load(pickle_in)
    print("\nInductor Model")
    print('Model loaded from file')

#Generates a model and writes to a file if no file exists prior
else:
    #Code used for model training, 1k iterations
    best = 0
    for _ in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3)
        linear = linear_model.LinearRegression()

        #How accurate the model is
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print(best, " ", acc)

        if acc > best:
            best = acc
            #Saves the model to a file
            with open("matchingmodell.pickle", "wb") as f:
                pickle.dump(linear, f)
    print("\nInductor Model")

#print(best)
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


#Plotting the outputs //should we use exact values
'''
p = "c"
style.use("ggplot")
pyplot.scatter(data[p], data["input_power"])
pyplot.xlabel(p)
pyplot.ylabel("input_power")
pyplot.show()
'''