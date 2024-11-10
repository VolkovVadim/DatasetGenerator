import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


EXAMPLES_COUNT     = 5000
FUNC_TYPE          = 3


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


def visualize(data: pd.DataFrame) -> None:
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 10), dpi=80)

    label_font = {
        'family': 'serif',
        'color': 'darkblue',
        'size': 10
    }

    points_count = data.shape[0]

    plt.xlabel('Feature 1', fontdict=label_font)
    plt.ylabel('Feature 2', fontdict=label_font)
    plt.title(f"Multilabel classification ({points_count} points)")

    color_map = {
        0: 'blue',
        1: 'lime',
        2: 'green'
    }

    color = [color_map[label] for label in data.class_label]
    point_size = 5 if points_count < 20000 else 1

    plt.scatter(
        data.feature_1,
        data.feature_2,
        s=point_size,
        c=color
    )

    plt.show()


def generate(func_type: int = 1) -> pd.DataFrame:
    if func_type < 1 or func_type > 3:
        func_type = 1

    min_f1, max_f1 = -25.0, 25.0
    min_f2, max_f2 = -25.0, 25.0

    F1 = np.random.uniform(min_f1, max_f1, size=(EXAMPLES_COUNT,))
    F2 = np.random.uniform(min_f2, max_f2, size=(EXAMPLES_COUNT,))

    if func_type == 1:
        Y = []
        min_limit, max_limit = -10.0, 10.0
        for i in range(EXAMPLES_COUNT):
            diff = F1[i] - F2[i]
            if diff <= min_limit:
                Y.append(0)

            if diff > min_limit and diff <= max_limit:
                Y.append(1)

            if diff > max_limit:
                Y.append(2)


    if func_type == 2:
        Y = []
        for i in range(EXAMPLES_COUNT):
            min_limit, max_limit = -10.0, 10.0
            diff = F1[i] - 10.0 * np.sin(0.25 * F2[i])
            if diff <= min_limit:
                Y.append(0)

            if diff > min_limit and diff <= max_limit:
                Y.append(1)

            if diff > max_limit:
                Y.append(2)

    if func_type == 3:
        Y = []
        for i in range(EXAMPLES_COUNT):
            value_1 = F2[i] + 5.5 * np.sin(0.45 * F2[i]) - 15.0
            value_2 = F2[i] + 5.05* np.sin(0.95 * F2[i]) + 15.0

            if F1[i] <= value_1:
                Y.append(0)

            if F1[i] > value_1 and F1[i] <= value_2:
                Y.append(1)

            if F1[i] > value_2:
                Y.append(2)

    data = {
        "feature_1": F1,
        "feature_2": F2,
        "class_label": Y
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Versions")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Pandas : {pd.__version__}")

    # Parse command args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save genereated dataset to file", action="store_true")
    parser.add_argument("-c", "--count", help="dataset points count", type=int)
    parser.add_argument("-t", "--type", help="generator function type", type=int)

    args = parser.parse_args()

    if args.count:
        EXAMPLES_COUNT = args.count

    if args.type:
        FUNC_TYPE = args.type

    # Generate data
    df_generated_dataset = generate(FUNC_TYPE)
    show_info(df_generated_dataset)
    visualize(df_generated_dataset)

    # Save data to file
    if args.save:
        dataset_filename = f"multilabel_classification_v{FUNC_TYPE}_{EXAMPLES_COUNT}.csv"
        df_generated_dataset.to_csv(dataset_filename, index=False)
        print(f"Dataset saved to file : {dataset_filename}")

    print("Success")
