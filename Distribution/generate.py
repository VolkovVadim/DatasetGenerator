import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from matplotlib import colors


VALUES_COUNT=10000
BINS_COUNT=100


def visualize(values: np.array, bins: int) -> None:
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 10), dpi=80)

    N, bins, patches = plt.hist(
        values,
        bins=bins,
        color='skyblue',
        edgecolor='black'
    )


    # Setting color
    fracs = ((N**(1 / 5)) / N.max())
    norm = colors.Normalize(fracs.max() / 2, fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        #color = plt.cm.viridis(norm(thisfrac))
        #color = plt.cm.winter(norm(thisfrac))
        #color = plt.cm.YlGn(norm(thisfrac))
        color = plt.cm.autumn(norm(thisfrac))
        thispatch.set_facecolor(color)


    # Draw labels
    label_font = {
        'family': 'serif',
        'color': 'darkblue',
        'size': 14
    }

    plt.xlabel("Values", fontdict=label_font, labelpad=15)
    plt.ylabel("Frequency", fontdict=label_font, labelpad=15)
    plt.title(f"Distribution (values count : {len(values)})")


    plt.show()


def show_info(dataframe: pd.DataFrame) -> None:
    print(f"Dataframe :\n{dataframe.head(20)}\n")
    print(f"Records count    : {dataframe.shape[0]}\n")
    print(f"NaN values count :\n{dataframe.isna().sum()}")

    column_names = list(dataframe)
    max_len = len(max(column_names, key=len))
    if max_len < 10:
        max_len = 10

    print(f"Data types :")
    for column_name, data_type in dataframe.dtypes.items():
        print(f"  {column_name}{' ' * (max_len - len(column_name))} : <{data_type}>")

    print("\n")


def generate(count: int):
    #min_value, max_value = 1.0, 1000.0
    #values = np.random.uniform(min_value, max_value, size=(count,))

    loc, scale = 100, 10
    values = np.random.normal(loc, scale, size=(count,))

    return values




if __name__ == "__main__":
    print("Versions")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Pandas : {pd.__version__}")

    # Parse command args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save genereated dataset to file", action="store_true")
    parser.add_argument("-c", "--count", help="dataset points count", type=int)
    parser.add_argument("-b", "--bins", help="histogram bins count", type=int)

    args = parser.parse_args()

    if args.count:
        VALUES_COUNT = args.count

    if args.bins:
        BINS_COUNT = args.bins

    # Generate
    data = generate(VALUES_COUNT)
    visualize(data, BINS_COUNT)


    print("\nSuccess\n")