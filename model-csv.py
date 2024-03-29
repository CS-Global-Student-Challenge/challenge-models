# Modals
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
# Test train split
from sklearn.model_selection import train_test_split
import pandas as pd
# Error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv

# Reading the data.
fileName = "./train_data.csv"
data = pd.read_csv(fileName)

# Defing the X and Y.
X = data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = data['failed']
Z = data[['job_id','failed']]

# Splitting the data for training and testing.
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)


def DecisionTree():
    myModal = DecisionTreeRegressor()
    myModal.fit(train_x, train_y)
    predicated_y = myModal.predict(test_x)
    return Measure_Error(test_y, predicated_y)


def LinearReg():
    myModal = LinearRegression()
    myModal.fit(train_x, train_y)
    predicated_y = myModal.predict(test_x)
    return Measure_Error(test_y, predicated_y)


def LogisticReg():
    myModal = LogisticRegression()
    myModal.fit(train_x, train_y)
    predicated_y = myModal.predict(test_x)
    return Measure_Error(test_y, predicated_y)


def RandomForest():
    myModal = RandomForestRegressor()
    myModal.fit(train_x, train_y)
    predicated_y = myModal.predict(test_x)
    return Measure_Error(test_y, predicated_y)


def AdaBoost():
    myModal = AdaBoostClassifier()
    myModal.fit(train_x, train_y)
    predicated_y = myModal.predict(test_x)
    return Measure_Error(test_y, predicated_y)


def Measure_Error(test_y, predicated_y):
    return [mean_absolute_error(test_y, predicated_y), mean_squared_error(test_y, predicated_y)]


##### Print #####
print("Mean absolute Error (Lower === Better)")
print("Decision Tree:", DecisionTree())
print("Linear Regression:", LinearReg())
print("Logistic Regression:", LogisticReg())
print("Random Forest:", RandomForest())
print("AdaBoost:", AdaBoost())

#### Copying data from one csv file to other #####
filename1 = "model_complete_test.csv"
with open(filename1,'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(Z)

# #### Merging Data from one csv file to other####
# a = pd.read_csv("train_data.csv")
# b = pd.read_csv("model_complete_test.csv")
# b = b.dropna(axis=1)
# merged = a.merge(b, on='job_id')
# merged.to_csv("output.csv", index=False)
