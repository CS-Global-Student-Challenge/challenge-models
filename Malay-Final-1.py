import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Reading the file
data = pd.read_csv('./train_data.csv', index_col="job_id")

# Defing the X and Y.
X = data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = data['failed']

# Splitting the data for training and testing.
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model
myModal = KNeighborsClassifier()
myModal.fit(train_x, train_y)
predicated_y = myModal.predict(test_x)


# Getting the accuracy.
print(
    mean_absolute_error(test_y, predicated_y))
# Writing to csv
output = pd.DataFrame({'job_id': test_x.index,
                       'failed': predicated_y})
output.to_csv('model_complete_test.csv', index=False)

score = accuracy_score(test_y, predicated_y)
print(score)


#### Output generates only 4000 rows #####
#### We need atleast 10000 rows for submission #####
