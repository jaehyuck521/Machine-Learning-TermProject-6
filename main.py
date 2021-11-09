import pandas as pd
import numpy as np
# row 생략 없이 출력
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype

from check_dataset import check_dataset

pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    df = pd.read_csv(r".\dataset\vehicles.csv")


    # df = pd.read_csv(r".\dataset\used_car_filtered.csv")
    # df_drop_outlier = check_dataset(df.copy())

