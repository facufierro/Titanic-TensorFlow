# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython

# ----------1. LOAD DATASET----------#
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
# print(dftrain.head())
# remove column 'survived' from deftrain and store it in y_train
y_train = dftrain.pop('survived')
# remove column 'survived' from defeval and store it in y_eval
y_eval = dfeval.pop('survived')

# show a graph for the age of the passengers
# dftrain.age.hist(bins=20)
# show a graph with percetage of passengers survival rate by sex
# pd.concat([dftrain, y_train], axis=1).groupby(
#     'sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# ----------2. SEPARATE AND MODIFY CATEGORICAL COLUMNS----------#
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses',
                       'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # create a vocabular for each feature
    vocabulary = dftrain[feature_name].unique()
    # create a categorical column, and store it in feature_columns
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))
    # create a numeric column, and store it in feature_columns
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))

# print(feature_columns)

# ----------3. CREATE THE INPUT FUNCTION----------#
# create a function that returns a tf.data.Dataset object for the training data
# shuffle the data and repeat it for the number of epochs
# batch the data into batches of 32
# return the dataset


def make_input_fn(data_df, label_df, num_epochs=500, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use


# call the input_function that was returned to us to get a dataset object we can feed to the model
train_input_fn = make_input_fn(dftrain, y_train)
# do the same for the test set
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# ----------4. CREATE THE MODEL----------#
# create a linear estimator by passing the feature columns created earlier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# train the model
linear_est.train(train_input_fn)
# evaluate the model
result = linear_est.evaluate(eval_input_fn)
# the result variable is simply a dict of stats about our model
# print(result['accuracy'])

# ----------5. PREDICT----------#
# get a prediction from the model
result = list(linear_est.predict(eval_input_fn))

# ask the user for the index of the passenger
i = int(input('Enter the index of the passenger: '))
# print the passenger's data as a data grid with column names
IPython.display.clear_output(wait=True)
print("--------------------------------------------------")
print(dfeval.loc[i].to_frame().T)
if (print(y_eval.loc[i]) == 1):
    print("Survived")
else:
    print("Died")
# print survival chance in percentage with 2 decimal places
print("--------------------------------------------------")
print('Survival chance: {:.2f}%'.format(
    result[i]['probabilities'][1]*100))
print("--------------------------------------------------")


# %%
