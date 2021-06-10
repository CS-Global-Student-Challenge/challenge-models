# Modals
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
# Test train split
from sklearn.model_selection import train_test_split
import pandas as pd
# Error
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reading the data.
fileName = "./train_data.csv"
data = pd.read_csv(fileName, index_col="job_id")

# Defing the X and Y.
X = data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = data['failed']

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


def KNN(n):
    myModal = KNeighborsClassifier(n_neighbors=n)
    myModal.fit(train_x, train_y)
    predicated_y = myModal.predict(test_x)
    return Measure_Error(test_y, predicated_y)


def Measure_Error(test_y, predicated_y):
    return [mean_absolute_error(test_y, predicated_y), mean_squared_error(test_y, predicated_y)]


##### Print #####
print("Mean absolute Error (Lower === Better)")
# print("Decision Tree:", DecisionTree())
# print("Linear Regression:", LinearReg())
# print("Logistic Regression:", LogisticReg())
# print("Random Forest:", RandomForest())
print("AdaBoost:", AdaBoost())
print("KNN: 5 ", KNN(5))
print("KNN: 4 ", KNN(4))
print("KNN: 3 ", KNN(3))
