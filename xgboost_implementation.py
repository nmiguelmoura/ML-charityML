import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, fbeta_score

# import xgboost algorithm
# This was installed via anaconda
# >>>> conda install -c msarahan py-xgboost
from xgboost import XGBClassifier

data = pd.read_csv("census.csv")

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis=1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(
    lambda x: np.log(x + 1))


# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()  # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss',
             'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(
    features_log_transformed[numerical])

# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
categorical_data = ["workclass", "education_level", "marital-status",
                    "occupation", "relationship", "native-country", "race",
                    "sex"]
features_final = pd.get_dummies(data=features_log_minmax_transform,
                                columns=categorical_data)

# Encode the 'income_raw' data to numerical values
income = income_raw.replace(['<=50K', '>50K'], [0, 1])

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    stratify=income,
                                                    test_size=0.2,
                                                    random_state=0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


######## XGBOOST IMPLEMENTATION #######

# Instantiate and fit the model using training data.
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Predict results using testing data.
predictions = clf.predict(X_test)

# Calculate accuracy and f_score.
accuracy = accuracy_score(y_test, predictions)
f_score = fbeta_score(y_test, predictions, 0.5)
print "testing set accuracy: ", accuracy
print "testing set f-score: ", f_score

# Even using the model without any tweaks, the result is already better than the
# one achieved with tweaked Decision Tree Classifier.
# However the model takes more time to fit the data. Even so, I would prefer
# this algorithm to Decision Tree Classifier.

