from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier as ABC
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, neighbors
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn import svm
def feature_selection(df):


    # df['Type 2'] = pd.DataFrame(le.fit_transform(df['Type 2'].astype(str)))
    # print(df)

    # print(df)
    target_names = np.array([1,0])

    # df = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    # df = pd.read_csv('iris.data.txt', header= None, names = names)
    # print(df)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    # print(df)
    # df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    # print(df)
    # target_names = np.array(['setosa', 'versicolor', 'virginica'])
    # features = np.array(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    # features = df.columns[1:14]
    # print(features)
    features = np.array(['satisfaction_level',
                         'last_evaluation',
                         'number_project',
                         'average_montly_hours',
                         'time_spend_company',
                         'Work_accident',
                         'promotion_last_5years',
                         'sales',
                         'salary'])
    # features = np.delete(features,[0])
    # print(features)
    k = 1
    features_record = []
    acc = []
    while len(features) > k:
        features_record.append(features)
        forest = RFC(n_jobs=2, n_estimators=150)
        y = train['left']
        forest.fit(train[features], y)
        acc.append(forest.score(test[features], test['left']))
        #
        preds = target_names[forest.predict(test[features])]
        # print(preds)
        # print(test['left'])
        # print (pd.crosstab(index=test['left'], columns=preds, rownames=['actual'], colnames=['preds']))
        #
        importances = forest.feature_importances_
        # print(importances)
        indices = np.argsort(importances)
        # print(indices[0])

        # indices = np.delete(indices, [0])
        # print(features)
        # print(indices)
        #
        #
        # plt.figure(1)
        # plt.title('Feature Importances')
        # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        # plt.yticks(range(len(indices)), features[indices])
        # plt.xlabel('Relative Importance')
        # plt.show()
        features = np.delete(features, [indices[0]])
    print(acc)

    features = features_record[acc.index(max(acc))]
    # print(features)
    return features
def classify(df, feature_name,type):
    # convert to numpy array
    x = df[feature_name].values
    y = df['left'].values
    skf = StratifiedKFold(n_splits=10,random_state=2, shuffle=True)
    # skf.get_n_splits(x, y)
    fig = plt.figure()

    plot_data = []

    test_acc = []
    val_acc = []
    for train_index, test_index in skf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        inner_skf = StratifiedKFold(n_splits=10)
        acc = []

        # in_count = [0,0,0,0]
        in_y_true = []
        in_y_pred = []
        y_true = []
        y_pred = []
        models = []
        for in_train_index, in_val_index in inner_skf.split(x_train, y_train):
            in_x_train, in_x_val = x_train[in_train_index], x_train[in_val_index]
            in_y_train, in_y_val = y_train[in_train_index], y_train[in_val_index]

            if type == 'Decision Tree':
                classifier = tree.DecisionTreeClassifier()
            elif type == 'k-NN':
                classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
            elif type == 'Random Forests':
                classifier = RFC(n_jobs=2, n_estimators=150)
            elif type == 'AdaBoost':
                classifier = ABC(n_estimators=150)
            elif type == 'SVM':
                classifier = svm.SVC(kernel='linear')


            classifier.fit(in_x_train, in_y_train)
            in_y_true += in_y_val.tolist()
            in_y_pred += classifier.predict(in_x_val).tolist()
            acc.append(classifier.score(in_x_val, in_y_val))
            models.append(classifier)
            # print(acc)
        model_choose = models[acc.index(max(acc))]
        acc = np.array(acc)
        print('train matrix')
        icf = confusion_matrix(in_y_true, in_y_pred, labels=[True, False])
        print(icf)
        i_precision = icf[0][0] / sum(icf[0])
        # print(i_precision)
        i_recall = icf[0][0] / (icf[0][0] + icf[1][0])
        print(i_recall)
        i_accuracy = np.mean(acc)
        val_acc.append(i_accuracy)
        i_F = (i_precision*i_recall / (i_precision+i_recall))*2
        i_TNR = icf[1][1] / (icf[1][1] + icf[0][1])

        print('Precision\tRecall\tAccuracy\tF\tTrue Negative Rate\n' + str(i_precision)+'\t'+str(i_recall) + '\t' + str(i_accuracy) + '\t' + str(i_F) + '\t' + str(i_TNR))
        # plt.subplot(2,5,count_plot)
        plot_data.append(np.array(acc))

        y_true += y_test.tolist()
        y_pred += model_choose.predict(x_test).tolist()
        test_acc.append(model_choose.score(x_test, y_test))
        # plt.show()
        # print (pd.crosstab(index=y_true, columns=y_pred, rownames=['actual'], colnames=['preds']))
        print('test matrix')
        cf = confusion_matrix(y_true, y_pred, labels = [True, False])
        print(cf)
        precision = cf[0][0] / sum(cf[0])
        recall = cf[0][0] / (cf[0][0] + cf[1][0])
        accuracy = test_acc[-1]
        F = 2 * precision * recall / (precision + recall)
        TNR = cf[1][1] / (cf[1][1] + cf[0][1])

        print('Precision\tRecall\tAccuracy\tF\tTrue Negative Rate\n' + str(precision) + '\t' + str(recall) + '\t' + str(
            accuracy) + '\t' + str(F) + '\t' + str(TNR))

    labels = list([1,2,3,4,5,6,7,8,9,10])
    print('val_mean_acc:' + str(np.mean(val_acc)) + '\t' + 'test_mean_Acc:' + str(np.mean(test_acc)))
    plt.boxplot(np.array(plot_data), labels=labels)
    plt.title( type + ' - Train data box plot')
    plt.show()

    plt.boxplot(np.array(test_acc))
    plt.title(type+' - Test data box plot')
    plt.show()
if __name__ == '__main__':
    df = pd.read_csv('HR_comma_sep.csv')
    le = LabelEncoder()
    for i, col in enumerate(df):
        if df[col].dtypes == 'object':
            # use astype avoid null value in col Type 2
            df[col] = pd.DataFrame(le.fit_transform(df[col].astype(str)))
    target_names = np.array([False, True])
    features = feature_selection(df)
    print(features)
    # features = np.array(['satisfaction_level',
    #                     'last_evaluation',
    #                     'number_project',
    #                     'average_montly_hours',
    #                     'time_spend_company',
    #                     'Work_accident',
    #                     'promotion_last_5years',
    #                     'sales',
    #                      'salary'])
    types = ['Decision Tree', 'k-NN', 'Random Forests', 'AdaBoost', 'SVM']
    for v in types:
        print(v)
        classify(df,features, v)
    # classify(df,features, 'Random Forests')