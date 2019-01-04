import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

df = pd.read_csv('model/mnist_dataset.csv', index_col=0)
mnist_data = df.iloc[:,1:]
mnist_target = df.iloc[:,0]

clf = LogisticRegression(solver='saga', tol=0.1, multi_class='multinomial')
clf.fit(mnist_data, mnist_target)

# joblib.dump(clf, 'model/logistic_regression.pkl')
pickle.dump(clf, open('model/logistic_regression.pkl', 'wb'))