import pandas as pd
import numpy as np

from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)


def get_numeric_col(df):
    numeric_col_list = []

    for col_name in df.columns:
        if is_numeric_dtype(df[col_name].dtypes):
            numeric_col_list.append(col_name)

    return numeric_col_list


def get_string_col(df):
    string_col_list = []

    for col_name in df.columns:
        if is_string_dtype(df[col_name].dtypes):
            string_col_list.append(col_name)

    return string_col_list


def outlier_iqr(df):
    numeric_col_list = get_numeric_col(df)
    total_size = df.shape[0]
    total_outlier_size = 0
    for col_name in numeric_col_list:
        q1, q3 = np.percentile(df[col_name], [25, 75])

        iqr = q3 - q1

        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)

        count = 0
        count += df[upper_bound < df[col_name]].shape[0]
        count += df[df[col_name] < lower_bound].shape[0]

        if count < total_size * 0.025:
            df = df[upper_bound >= df[col_name]]
            df = df[df[col_name] >= lower_bound]
            total_outlier_size += count

    return total_outlier_size, df


def check_dataset(df):
    print(df.info())
    print()
    print(f'number of "NULL" value: {df.isnull().sum().sum()}')
    df_drop_NAN = df.dropna(axis=0)
    print(f'droped row : {df.shape[0] -df_drop_NAN.shape[0]}', end='\n\n')

    for col_name in get_string_col(df_drop_NAN):
        if df_drop_NAN[col_name].dtype == 'object':
            print(f'{col_name}(categorical column) has "{len(df_drop_NAN[col_name].unique())}" different value')

    print()
    num_outlier, df_drop_outlier = outlier_iqr(df_drop_NAN.copy())
    print(f'number of outlier : {num_outlier} / {df_drop_NAN.shape[0]} ---> {df_drop_NAN.shape[0] - num_outlier}')

    return df_drop_outlier


if __name__ == "__main__":
    df = pd.read_csv(r'.\dataset\vehicles.csv')
    df.drop(['county', 'url', 'image_url', 'description', 'posting_date'], axis=1, inplace=True)
    df_drop_outlier = check_dataset(df.copy())
    print(df_drop_outlier['year'].value_counts())


    # df_drop_outlier.to_csv(r".\dataset\used_car_filtered.csv")

    # df_drop_NAN = df.dropna(axis=0)
    # print(f'number of row that has "null" value : {df.shape[0] - df_drop_NAN.shape[0]}', end='\n\n')
