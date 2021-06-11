import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsTransformer,NearestCentroid
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Reading the file
data = pd.read_csv('./train_data.csv', index_col="job_id")
dataTest = pd.read_csv("./test_data_unlabeled.csv", index_col="job_id")
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
output.to_csv('model_complete_test.csv', index=False)

