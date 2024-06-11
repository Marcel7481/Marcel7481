import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# print(train_data.iloc[0:4])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.iloc[-4:-1,[1,-2,-1]])
# na_counts = all_features.isna().sum()
# na_columns = na_counts[na_counts > 0]
# print(na_columns)
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std())
print(all_features.isna().sum())