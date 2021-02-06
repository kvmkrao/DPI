# Author: Murali 
# Date modified : Feb 6, 2021
# Sample code 
"""DNNRegressor"""
import sys 
import itertools
import pandas as pd
import tensorflow as tf
import numpy as np 

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.utils import shuffle
from subprocess import call

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS =  ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "resf"]
FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14"]
LABEL = "resf"

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

def main1(training_set,test_set):
  global nTraining, MSE, varh;

  df  = shuffle(training_set); # randomize training data
  dft = test_set[FEATURES]; # randomize training data

  n = nTraining
  print('# of training data used = %d \n' % (n) )
  training_set1 = df[0:n];

  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  opti = tf.train.AdamOptimizer(learning_rate = 0.0001,epsilon=0.01)
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,hidden_units=[40, 20],activation_fn=tf.nn.sigmoid,dropout = 0.045,optimizer=opti,model_dir='/model')

  regressor.train(input_fn=get_input_fn(training_set1),steps=10000)
  ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  prediction = regressor.predict(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  y_predicted = np.array(list(p['predictions'] for p in prediction))
  print(y_predicted)

  dft = dft.assign(y_predicted=y_predicted)
  print(len(y_predicted))
  print(len(dft[FEATURES]))
  dft.to_csv("test_predict.csv", sep='\t', index=False)

  y_actual= test_set[LABEL].values
  score_sklearn = metrics.mean_squared_error(y_predicted, y_actual)
  print('MSE (sklearn): {0:f}'.format(score_sklearn))
  print('Variance score: %.2f' % r2_score(y_actual, y_predicted))

  pyplot.figure(figsize=[10,6])
  pyplot.plot(y_predicted, marker='o', linestyle='--',label='predicted value',color="blue",linewidth=2.0)
  pyplot.plot(y_actual, '*-',label='actual value',color="red",linewidth=2.0)
  pyplot.xlabel('data points',fontsize=16)
  pyplot.ylabel(' output',fontsize=16)
  pyplot.legend(loc='upper right', numpoints=1, ncol=2, fancybox=False, shadow=False,fontsize=16)
  pyplot.savefig('predict.png')
  pyplot.show()
  pyplot.close()

  del df 
  del training_set 
  return;  

#-----------------------------------
global nTraining, MSE, varh;
nTraining = 5; MSE = 0; varh = 0; ntrials =1;
training_set = pd.read_csv("training_norm.dat", skipinitialspace=True,skiprows=1, names=COLUMNS,delim_whitespace=True)
test_set = pd.read_csv("testing_norm.dat", skipinitialspace=True,skiprows=1, names=COLUMNS,delim_whitespace=True)

f=open('output_trials.csv','w+')
for i in np.arange(219,220,1):
  for j in range(ntrials):
    nTraining = i;
    main1(training_set,test_set)
    print('%d %d %f %f \n' % (nTraining, j+1, MSE, varh) )
    f.write('%d %d %f %f \n' % (nTraining, j+1, MSE, varh) )
f.close();

