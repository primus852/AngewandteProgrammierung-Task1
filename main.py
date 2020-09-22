# Misc
import argparse
import operator

# Math Stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

# Add Models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Define Headers according to
headings = ['age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'martial-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'income'
            ]


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        rank_one_seen = False
        rank_two_seen = False
        rank_three_seen = False
        for candidate in candidates:
            if i < 4:
                if i == 1 and rank_one_seen is False:
                    print("Model with rank: {0}".format(i))
                    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
                    print("Parameters: {0}".format(results['params'][candidate]))
                    print("")
                    rank_one_seen = True
                if i == 2 and rank_two_seen is False:
                    print("Model with rank: {0}".format(i))
                    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
                    print("Parameters: {0}".format(results['params'][candidate]))
                    print("")
                    rank_two_seen = True
                if i == 3 and rank_three_seen is False:
                    print("Model with rank: {0}".format(i))
                    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
                    print("Parameters: {0}".format(results['params'][candidate]))
                    print("")
                    rank_three_seen = True


def check_unique(col):
    return col.unique()


def encode(data, with_class=True):
    encoder = LabelEncoder()

    for col in data.columns:
        if not with_class:
            if col != "income":
                data[col] = encoder.fit_transform(data[col])
        else:
            data[col] = encoder.fit_transform(data[col])

    return data


def show_dropped(org, dropped):
    count_cleaned = len(dropped)
    count_all = len(org)
    unclean_pct = 100 - (100 * count_cleaned / count_all)

    return unclean_pct, (count_all - count_cleaned)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--missing', default='ignore', choices=['drop', 'replace', 'ignore'],
                        help='How to handle missing data')

    parser.add_argument('--plots', default='display', choices=['display', 'save', 'both'],
                        help='Display plots immediately, save them or both')

    parser.add_argument('--optimize', type=bool, default=False,
                        help='Optimize via RandomGridSearchCV')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for the probability predict')

    args = parser.parse_args()
    if args.missing != 'drop' and args.missing != 'replace' and args.missing != 'ignore':
        print('Please select either drop, replace or ignore for --missing')
        exit()

    # Import the Dataframe
    df_raw = pd.read_csv('data/adult.data', delimiter=', ', names=headings, converters={"header": float})
    print(df_raw.head())

    # Final Weight serves no purpose? DROP IT
    df = df_raw.drop(["fnlwgt"], axis=1)

    # Define the target
    Y = df['income']

    # Show the distribution of the target
    dist_target = df.groupby('income').size()

    # Plot the distribution
    plt.bar([1, 2], dist_target, tick_label=['<=50K', '>50K'], width=0.75, color=['#477ca8', '#cb3335'])
    plt.ylabel('Anzahl')
    plt.title('Verteilung des Targets (n=%s)' % len(df.index))

    if args.plots == 'display' or args.plots == 'both':
        plt.show()

    if args.plots == 'save' or args.plots == 'both':
        plt.savefig('./plots/target_dist.png')

    plt.close()

    # Encode the DF #
    df_orig = df.copy()

    # Get the unique values per column
    cols_with_missing = []
    print('\n------UNIQUE VALUES PER FEATURE------')
    for head in headings:
        if head != 'fnlwgt':
            unique = check_unique(df[head])
            print('Column: %s, Values: %s' % (head, unique))

            if '?' in unique:
                cols_with_missing.append(head)

    # Go through all columns with missing values
    print('\n------MISSING DATA------')
    for miss in cols_with_missing:
        occurrence = df[miss].value_counts(dropna=False)
        print(
            'Missing data in %s instances in %s, most occurrence for non-missing: %s' %
            (occurrence["?"], miss, max(occurrence.items(), key=operator.itemgetter(1))[0])
        )

    # Depending on the cli arg, do sth with the DataFrame
    if args.missing == 'drop':
        df_cleaned = df.replace(to_replace=['?'], value=np.nan)
        df_cleaned = df_cleaned.dropna()

        # Show percentage dropped
        pct_dropped, total_dropped = show_dropped(df, df_cleaned)
        print('%i (%.2f%%) instances dropped due to missing data' % (total_dropped, pct_dropped))

        df = df_cleaned.copy()

    elif args.missing == 'replace':
        df.replace(to_replace=['?'], value='b', inplace=True)

    unique_counts = []
    for head in headings:
        if head != 'fnlwgt':
            # Add count of features
            occurrences = df[head].value_counts(dropna=False)
            unique_counts.append(occurrences)

    # Go through all columns and check for unbalanced Features
    print('\n------FEATURE BALANCE------')
    for col in unique_counts:
        most_name = max(col.items(), key=operator.itemgetter(1))[0]
        most_value = max(col.items(), key=operator.itemgetter(1))[1]
        rep_pct = (most_value * 100 / len(df.index))
        print(
            'Most represented Value for %s: %s (%s / %.2f%%)' % (col.name, most_name, most_value, rep_pct)
        )

    # Encode the chars to integer in order to be able to process the data for the violin plot
    # DO NOT USE FOR MODEL!
    df_violin = encode(df, False)

    # Create a violin Plot to see the distributions of the attributes for the target #
    df_div = pd.melt(df_violin, "income", var_name="Features")
    fig, ax = plt.subplots(figsize=(10, 5))
    p = sns.violinplot(ax=ax, x="Features", y="value", hue="income", split=True, data=df_div, inner='quartile',
                       palette='Set1')
    df_no_income = df_violin.drop(["income"], axis=1)
    p.set_xticklabels(rotation=90, labels=list(df_no_income.columns))

    if args.plots == 'display' or args.plots == 'both':
        plt.show()

    if args.plots == 'save' or args.plots == 'both':
        plt.savefig('./plots/target_violin.png')

    plt.close()

    df_box = encode(df.copy())

    # Boxplot for params
    for head in headings:
        if head != 'income' and head != 'fnlwgt':
            ax = sns.boxplot(x="income", y=head, data=df_box, palette='Set1')

            if args.plots == 'display' or args.plots == 'both':
                plt.show()

            if args.plots == 'save' or args.plots == 'both':
                plt.savefig('./plots/box-%s.png' % head)

            plt.close()

    # Copy DataFrame for OneHotEncoding and drop the target
    df_oh_temp = df_orig.copy()
    df_oh = df_oh_temp.drop(["income"], axis=1)

    # OneHotEncoded DF
    X = pd.get_dummies(df_oh)

    # List of Classifiers we go through
    classifiers = {
        #"GradientBoosting": GradientBoostingClassifier(),
        #"kNN": KNeighborsClassifier(),
        #"kNN_opt": KNeighborsClassifier(n_neighbors=45, weights='distance', algorithm='brute', p=1),
        #"SVM": SVC(gamma='auto', probability=True),
        #"DecisionTree": DecisionTreeClassifier(max_leaf_nodes=8),
        "XGBoost": XGBClassifier(),
        #"RandomForest": RandomForestClassifier(n_estimators=10),
        #"RandomForest_opt": RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy', max_depth=11, max_features=10, min_samples_split=10),
        #"NeuralNet": MLPClassifier(alpha=1),
        #"NaiveBayes": GaussianNB(),
    }

    # RandomCV RandomForest
    param_dist_rf = {"max_depth": sp_randint(3, 20),
                     "max_features": sp_randint(1, 11),
                     "min_samples_split": sp_randint(2, 11),
                     "bootstrap": [True, False],
                     "criterion": ["gini", "entropy"]}

    param_dist_knn = {"n_neighbors": sp_randint(1, 50),
                      "weights": ['uniform', 'distance'],
                      "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      "p": sp_randint(1, 2)}

    # Split the Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Init Vars
    results = {}
    mean_auc = 0
    mean_acc = 0
    mean_score = 0
    total_time = 0

    print('\n------CLASSIFIER RESULTS------')
    # Go through all Classifiers
    for name, clf in list(classifiers.items()):
        # Start the Timer per Model
        start_timer = time.process_time()

        # Cross Validation Scores (KFold=5)
        scores = cross_val_score(clf, X, Y, cv=5)

        if name == 'RandomForest' and args.optimize is True:
            print('\n------RANDOM FOREST OPTIMIZATION------')
            # run randomized search
            n_iter_search = 10
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist_rf, n_iter=n_iter_search, cv=5)
            start_rf = time.process_time()
            random_search.fit(X_train, y_train)
            print("RandomizedSearchCV (RF) took %.2f seconds for %d candidates parameter settings." % (
                (time.process_time() - start_rf), n_iter_search))
            report(random_search.cv_results_)

        if name == 'kNN' and args.optimize is True:
            print('\n------KNN FOREST OPTIMIZATION------')
            # run randomized search
            n_iter_search = 10
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist_knn, n_iter=n_iter_search, cv=5)
            start_rf = time.process_time()
            random_search.fit(X_train, y_train)
            print("RandomizedSearchCV (knn) took %.2f seconds for %d candidates parameter settings." % (
                (time.process_time() - start_rf), n_iter_search))
            report(random_search.cv_results_)

        # Train with Split
        clf.fit(X_train, y_train)

        # Stop the timer
        end_timer = time.process_time()

        # Get Total Time
        clf_time = end_timer - start_timer

        if name != 'kNN':
            # get importance
            importance = clf.feature_importances_
            # summarize feature importance
            for i, v in enumerate(importance):
                print('Feature: %0d, Score: %.5f' % (i, v))
            # plot feature importance
            plt.bar([x for x in range(len(importance))], importance)

            if args.plots == 'save' or args.plots == 'both':
                plt.savefig('./plots/fi-{}.png'.format(name))

        # Predict with the Testset
        clf_pred = clf.predict(X_test)

        # Mean of CrossVal
        mean = scores.mean()

        # Get the Accuracy for the Test Dataset
        test_accuracy = accuracy_score(y_test, clf_pred)

        probs = clf.predict_proba(X_test)
        preds = probs[:, 1]

        fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=">50K", drop_intermediate=False)
        roc_auc = metrics.auc(fpr, tpr)

        # Add to Mean AUC
        mean_auc += roc_auc

        # Add to Mean Score
        mean_score += mean

        total_time += clf_time

        print('---Plotting ROC ({})---'.format(name))

        # Plot the ROC
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='%s (AUC:%0.3f)' % (name, roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        if args.plots == 'display' or args.plots == 'both':
            plt.show()

        if args.plots == 'save' or args.plots == 'both':
            plt.savefig('./plots/roc.png')

        plt.close()

        # Confusion Matrix with a custom Threshold
        predict_thresh = np.where(preds > float(args.threshold), ">50K", "<=50K")
        confusion_matrix = metrics.confusion_matrix(y_test, predict_thresh)

        # Plot the Optimal Threshold
        current_t_val = []
        false_negatives = []
        current_threshold = 0.00
        first_time = None
        print('---Plotting Threshold ({})---'.format(name))
        # Plot a Line that shows the FN-Rate --> so if FN stars to grow, use this TH
        # Tells us "how sure" the Model should be
        while current_threshold <= 1:
            current_threshold += 0.001
            predict_thresh = np.where(preds > current_threshold, ">50K", "<=50K")
            confusion_matrix_plot = metrics.confusion_matrix(y_test, predict_thresh)
            current_t_val.append(current_threshold)
            false_negatives.append(confusion_matrix_plot[1][0])
            if first_time is None and confusion_matrix_plot[1][0] > 0:
                first_time = current_threshold

        t_data = {'Threshold Value': current_t_val, 'FN (>50K als <=50K klassifiziert)': false_negatives}
        sns_plot = sns.lineplot(x='Threshold Value', y='FN (>50K als <=50K klassifiziert)', data=pd.DataFrame(t_data), label='Max. %s: <%.3f' % (name, float(first_time)), palette='Set1')
        sns_fig = sns_plot.get_figure()

        if args.plots == 'save' or args.plots == 'both':
            sns_fig.savefig('./plots/th-%s.png' % name)

        # Visualization
        results[name] = {
            'classifier': name,
            'time': clf_time,
            'mean': mean,
            'accuracy': test_accuracy,
            'matrix': confusion_matrix,
            'auroc': roc_auc,
            'threshold': threshold
        }

    # Print all the results
    for name, result in list(results.items()):
        print('%s: Accuracy %0.2f | MeanCV %0.2f | Time %0.2f' %
              (result['classifier'], result['accuracy'], result['mean'], result['time']))
        print('CF Matrix')
        print(result['matrix'])
