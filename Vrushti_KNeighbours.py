import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Reading the file
train_data = pd.read_csv('train_data.csv', index_col="job_id")
test_data = pd.read_csv("test_data_unlabeled.csv", index_col="job_id")

# Define X and Y
X = train_data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = train_data['failed']
test_X = test_data[['memory_GB', 'network_log10_MBps',
                   'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]

# Splitting the data for training and testing.
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, train_size=0.5, test_size=0.5, random_state=0)

# Model
k_model = KNeighborsClassifier(n_neighbors=4)
k_model.fit(X, Y)
predicted_y = k_model.predict(test_X)

# Writing to csv
output = pd.DataFrame({'job_id': test_X.index,
                       'failed': predicted_y})
output.to_csv('submission.csv', index=False)

# Calculating accuracy
score = accuracy_score(test_y, predicted_y)
print(score)
