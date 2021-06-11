import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Reading the file
data = pd.read_csv('./train_data.csv', index_col="job_id")
dataTest = pd.read_csv("./test_data_unlabeled.csv", index_col="job_id")

# Defining X and Y
X = train_data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = train_data['failed']
test_X = test_data[['memory_GB', 'network_log10_MBps',
                   'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]

# Model
myModel = SVC(gamma=2.25, C=1)
myModel.fit(X, Y)
predicted_y = myModel.predict(test_X)

# Writing to csv
output = pd.DataFrame({'job_id': test_X.index,
                       'failed': predicted_y})
output.to_csv('model_complete_test.csv', index=False)

