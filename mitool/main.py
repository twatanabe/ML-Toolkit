from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from IPython.display import display
import missingno as msno
import plotly.express as px

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

def summary(train):
    sum = pd.DataFrame(train.dtypes, columns=['dtypes'])
    sum['missing#'] = train.isna().sum()
    sum['missing%'] = (train.isna().sum())/len(train)
    sum['uniques'] = train.nunique().values
    sum['count'] = train.count().values
    return sum

def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

def read_datasets(train_path: str, test_path: str):
    """
    Reads train and test datasets from csv files and returns them as pandas DataFrames.

    Parameters:
    - train_path (str): The path to the train dataset csv file.
    - test_path (str): The path to the test dataset csv file.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def analyze_dataframe(df):
    """
    Analyze a pandas DataFrame and provide a summary of its characteristics.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to analyze.

    Returns:
    None
    """
    print("DataFrame Information:")
    print("----------------------")
    display(df.info(verbose=True, show_counts=True))
    print("\n")
    
    print("DataFrame Values:")
    print("----------------------")
    display(df.head(5).T)
    print("\n")

    print("DataFrame Description:")
    print("----------------------")
    display(df.describe().T)
    print("\n")

    print("Number of Null Values:")
    print("----------------------")
    display(df.isnull().sum())
    print("\n")

    print("Number of Duplicated Rows:")
    print("--------------------------")
    display(df.duplicated().sum())
    print("\n")

    print("Number of Unique Values:")
    print("------------------------")
    display(df.nunique())
    print("\n")

    print("DataFrame Shape:")
    print("----------------")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame and print the number of duplicates found and removed.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - df_no_duplicates: DataFrame with duplicates removed
    """

    # Identify duplicates
    duplicates = df[df.duplicated()]

    # Print number of duplicates found and removed
    print(f"Number of duplicates found and removed: {len(duplicates)}")

    # Remove duplicates
    df_no_duplicates = df.drop_duplicates()

    return df_no_duplicates

def describe_with_style(df):
    display(df.describe().T\
            .style.bar(subset=['mean'], color=px.colors.qualitative.G10[0])\
            .background_gradient(subset=['std'], cmap='Greens')\
            .background_gradient(subset=['50%'], cmap='BuGn'))

def plot_missing_data(df):
    msno.matrix(df=df, figsize=(10,6), color=(0,.3,.3))

def plot_histogram(train, test, feats):
    for feat in feats:
        plt.figure(figsize=(12,3))
        ax1 = plt.subplot(1,2,1)
        train[feat].plot(kind='hist', bins=50, color='blue')
        plt.title(feat + ' / train')
        ax2 = plt.subplot(1,2,2, sharex=ax1)
        test[feat].plot(kind='hist', bins=50, color='green')
        plt.title(feat + ' / test')
        plt.show()

def plot_heatmap(df):
    corr = df.corr().round(1)
    plt.figure(figsize=(20,10))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
    plt.show()

def plot_distribution(train, test, feats, target_feat):
    for feat in feats:
        plt.figure(figsize=(12,4))
        ax1 = plt.subplot(1,2,1)
        sns.boxplot(data=train, x=target_feat, y=feat)
        plt.title(target_feat + ' vs ' + feat + ' / train')
        x1 = plt.subplot(1,2,2)
        sns.boxplot(data=test, y=feat)
        plt.title(feat + ' / test')
        plt.show()