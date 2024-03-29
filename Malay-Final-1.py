import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Reading the file
data = pd.read_csv('./train_data.csv', index_col="job_id")
dataTest = pd.read_csv("./test_data.csv", index_col="job_id")
# Defing the X and Y.
X = data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = data['failed']
Test_X = dataTest[['memory_GB', 'network_log10_MBps',
                   'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]

# Model
myModal = KNeighborsClassifier(n_neighbors=4)
myModal.fit(X, Y)
predicated_y = myModal.predict(Test_X)

# Writing to csv
output = pd.DataFrame({'job_d': Test_X.index,
                       'failed': predicated_y})
output.to_csv('submission.csv', index=False)
