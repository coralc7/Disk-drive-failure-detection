import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import linregress
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, classification_report, f1_score, precision_recall_curve, auc
import shap
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def dataset_initial_understanding(dataset):
    print(dataset.info())
    print('\n')
    print(dataset.head())
    print('\n')
    print('Missing values - Preliminary inspection')
    data_percentage_missing_values = dataset.isnull().sum() / len(dataset) * 100
    print(data_percentage_missing_values)
    # target
    dataset['fail'] = dataset['fail'].astype('int')


def plot_target_distribution(dataset):
    plt.figure()
    color_target = {0: 'blue', 1: 'orange'}
    plot = sns.countplot(x=dataset['fail'], palette=color_target)
    total = len(dataset['fail'])
    for patch in plot.patches:
        percentage = '{:.1f}%'.format(100 * (patch.get_height() / total))
        x = patch.get_x() + patch.get_width() / 3
        y = patch.get_y() + patch.get_height() + 10
        plot.annotate(percentage, (x, y), size=12)
    plt.title('The Failures distribution')
    not_fail_patch = mpatches.Patch(color='blue', label='Non-failed')
    fail_patch = mpatches.Patch(color='orange', label='Failed')
    plt.legend(handles=[not_fail_patch, fail_patch], bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('Fail')
    plt.ylabel('Count')


def fail_percentage_negative_values(dataset, col_name):
    negative_subset = dataset[dataset[col_name] < 0]
    aggregate_negative_subset = negative_subset.groupby(['d_id']).mean()
    fail_percentage_subset = np.round(aggregate_negative_subset['fail'].sum() / aggregate_negative_subset.shape[0], 3)
    print('The failures percentage among negative values of feature ' + col_name + ' is: ' + str(fail_percentage_subset))
    disks_id_without_fail_with_negative = negative_subset[negative_subset['fail'] == 0]['d_id'].unique()
    print('The number of disks without failures but with negative values of feature ' + col_name + ' is: ' + str(
        disks_id_without_fail_with_negative.size))
    return disks_id_without_fail_with_negative.tolist()


def replace_negative_vals_average(dataset, col_name):
    average = int(np.round(dataset[col_name][dataset[col_name] >= 0].mean()))
    mask = dataset[col_name] < 0
    dataset.loc[mask, col_name] = average
    return dataset


def plot_categorical_distribution(data, col_name):
    plot = sns.countplot(x=data)
    total = len(data)
    for patch in plot.patches:
        percentage = '{:.1f}%'.format(100 * (patch.get_height() / total))
        x = patch.get_x() + patch.get_width() / 6
        y = patch.get_y() + patch.get_height() + 0.5
        plot.annotate(percentage, (x, y), size=12)
    plt.title('The ' + col_name + ' distribution')
    plt.xlabel(col_name)
    plt.show()


def plot_hist(data, col_name):
    plt.figure()
    plt.hist(data)
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.title('The Histogram of ' + col_name)
    plt.grid(True)


def plot_relation_numeric_to_target(data, col_name):
    plt.figure()
    color_target = {0: 'blue', 1: 'orange'}
    sns.boxplot(x='fail', y=col_name, data=data, palette=color_target)
    plt.title('The relationship between failure and ' + col_name)
    not_fail_patch = mpatches.Patch(color='blue', label='Non-failed')
    fail_patch = mpatches.Patch(color='orange', label='Failed')
    plt.legend(handles=[not_fail_patch, fail_patch], bbox_to_anchor=(1.0, 1.0))
    plt.show()


def plot_relation_categorical_to_target(data, col_name):
    sns.catplot(x=col_name, hue='fail', data=data, kind='count', legend=False)
    plt.title('The relationship between failure and ' + col_name)
    plt.legend(['Non-failed', 'Failed'], bbox_to_anchor=(1.0, 1.0))
    plt.show()


def remove_disk_id(dataset, d_id):
    mask = dataset['d_id'] != d_id
    dataset = dataset.loc[mask, :]
    return dataset


def get_slope_intercept(dataset, timeline, col_name):
    slope_intercept_aggregate = dataset.loc[:, ['d_id', col_name]].groupby('d_id').apply(
        lambda x: pd.Series(linregress(timeline, x[col_name]))).rename(columns={
        0: 'slope',
        1: 'intercept',
        2: 'rvalue',
        3: 'pvalue',
        4: 'stderr'}).loc[:, ['slope', 'intercept']]
    return slope_intercept_aggregate


def add_statistic_feature(origin_data, statistic_data, statictic_name):
    for feature in statistic_data.columns:
        origin_data.insert(origin_data.shape[1], statictic_name + '_' + feature, statistic_data[feature])


def tuning_hyperparameters(search_cv, X_train, y_train):
    grid_result = search_cv.fit(X_train, y_train)
    """
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    """
    chosen_model = grid_result.best_estimator_
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return chosen_model


def fit_mode(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    return y_pred_test, y_pred_train, y_pred_probs


def evaluation(model, y_train, y_pred_train, y_test, y_pred_test, y_pred_probs):
    print('** Train evaluation **')
    print('The f1 score is ' + str(np.round(f1_score(y_train, y_pred_train), 2)))
    print('The Recall score is ' + str(np.round(recall_score(y_train, y_pred_train), 2)))
    print('The precision score is ' + str(np.round(precision_score(y_train, y_pred_train), 2)))

    print('** Test evaluation **')
    print('The f1 score is ' + str(np.round(f1_score(y_test, y_pred_test), 2)))
    print('The Recall score is ' + str(np.round(recall_score(y_test, y_pred_test), 2)))
    print('The precision score is ' + str(np.round(precision_score(y_test, y_pred_test), 2)))
    # metrics for each class
    print(classification_report(y_test, y_pred_test))
    # Precision-Recall Curve
    y_pred_probs
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    # calculate precision-recall AUC
    auc_model = auc(recall, precision)
    print('The precision-recall AUC is ' + str(np.round(auc_model, 2)))
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')
    ax.set_title('The Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title('The confusion matrix')
    plt.show()


def plot_top_feature_importance(model, X_train, top=10):
    plt.rcParams.update({'font.size': 14})
    importance_features = model.feature_importances_
    sorted_indices = np.argsort(importance_features)[::-1]
    top_features = sorted_indices[0:top]
    plt.title('Feature importance')
    plt.bar(range(top), importance_features[top_features], align='center')
    plt.xticks(range(top), X_train.columns[top_features], rotation=90)
    plt.tight_layout()
    plt.show()

def shap_plot(model, X_train):
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    # shap.summary_plot(shap_values, X_train, plot_type='bar')
    plt.figure(figsize=(15, 8))
    shap.summary_plot(shap_values[1], X_train)