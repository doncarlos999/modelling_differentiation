import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = pd.read_csv('../DESeq2_primed_only/vsd.csv', index_col=0)
info = pd.read_csv('../info.tsv', sep='\t')
data = data.T
info['type_name'] = info['condition'] + info['full_name']
# replace start of name so it matches data
# will need to modify this if I add naive/reprimed data
pat = r"(?P<name>[pnr]).*c"
info['type_name'] = info['type_name'].str.replace(pat, 'p_c')
info['type_name'] = info['type_name'].str.replace(r'\s', '')
info.index = info['type_name']
data = data.join(info['pax6_%'])
data = data.dropna()
target = data['pax6_%']
data = data.drop('pax6_%', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.4, random_state=0)

clf = svm.SVR(kernel='linear', C=1)
scores = cross_val_score(clf, data, target, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# not bad scores i also tried some trees but they were worse

unscaled_data = pd.read_csv('../DESeq2_primed_only/vsd.csv', index_col=0)
unscaled_data = unscaled_data.T
unscaled_data = unscaled_data.join(info['pax6_%'])
unscaled_data = unscaled_data.dropna()
unscaled_target = unscaled_data['pax6_%']
unscaled_data = unscaled_data.drop('pax6_%', axis=1)
scaler = StandardScaler().fit(unscaled_data)
scaled_data = scaler.transform(unscaled_data)
clf = svm.SVR(kernel='linear', C=1)
scores = cross_val_score(clf, scaled_data, target, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# worse scores


def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack(
        [top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(1, 1 + 2 * top_features),
        feature_names[top_coefficients],
        rotation=60, ha='right')
    plt.show()


classifier = clf.fit(X_train, y_train)
feature_names = np.array(X_train.columns)
plot_coefficients(classifier, feature_names)
