import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from typing import List


EXAMPLES_COUNT     = 5000       # Points count in dataset
FUNC_TYPE          = 3          # Dataset type
NOISE_ALPHA        = 0.8        # Part of the dataset with outliers
NOISE_LEVEL        = 0.9        # Data noise level


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


def visualize(data: pd.DataFrame, borders: List[pd.DataFrame] = None) -> None:
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

    if borders is not None:
        borders_count = len(borders)
        for i in range(borders_count):
            plt.plot(
                borders[i].x,
                borders[i].y,
                linestyle='--',
                linewidth=2,
                c='midnightblue',
            )

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


def generate(
        count: int, 
        func_type: int = 1, 
        noise: bool = False,
        noise_alpha: float = 0.1,
        noise_level: float = 1.0
) -> pd.DataFrame:
    if func_type < 1 or func_type > 3:
        func_type = 1

    min_f1, max_f1 = -25.0, 25.0
    min_f2, max_f2 = -25.0, 25.0

    F1 = np.random.uniform(min_f1, max_f1, size=(count,))
    F2 = np.random.uniform(min_f2, max_f2, size=(count,))

    if func_type == 1:
        Y = []
        min_limit, max_limit = -10.0, 10.0
        for i in range(count):
            diff = F1[i] - F2[i]
            if diff <= min_limit:
                Y.append(0)

            if diff > min_limit and diff <= max_limit:
                Y.append(1)

            if diff > max_limit:
                Y.append(2)


    if func_type == 2:
        Y = []
        for i in range(count):
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
        for i in range(count):
            value_1 = F2[i] + 5.5 * np.sin(0.45 * F2[i]) - 15.0
            value_2 = F2[i] + 5.0 * np.sin(0.95 * F2[i]) + 15.0

            if F1[i] <= value_1:
                Y.append(0)

            if F1[i] > value_1 and F1[i] <= value_2:
                Y.append(1)

            if F1[i] > value_2:
                Y.append(2)

    if noise:
        examples_with_noise = int(float(count) * noise_alpha)
        selected_examples = np.random.randint(count, size=examples_with_noise)
        for i in selected_examples:
            F1[i] += np.random.normal(scale=noise_level)
            F2[i] += np.random.normal(scale=noise_level)


    data = {
        "feature_1": F1,
        "feature_2": F2,
        "class_label": Y
    }

    return pd.DataFrame(data)


def get_borders(func_type: int = 1) -> List[pd.DataFrame]:
    if func_type < 1 or func_type > 3:
        func_type = 1

    min_x, max_x = -25.0, 25.0
    min_y, max_y = -25.0, 25.0

    Y = np.arange(min_y, max_y, 0.025)

    result = []

    X1, Y1, X2, Y2 = [], [], [], []
    for i in range(len(Y)):
        if func_type == 1:
            x1, y1 = Y[i] - 10.0, Y[i]
            x2, y2 = Y[i] + 10.0, Y[i]

        if func_type == 2:
            x1, y1 = 10.0 * np.sin(0.25 * Y[i]) - 10.0, Y[i]
            x2, y2 = 10.0 * np.sin(0.25 * Y[i]) + 10.0, Y[i]

        if func_type == 3:
            x1, y1 = Y[i] + 5.5 * np.sin(0.45 * Y[i]) - 15.0, Y[i]
            x2, y2 = Y[i] + 5.0 * np.sin(0.95 * Y[i]) + 15.0, Y[i]

        if x1 >= min_x and x1 <= max_x:
            X1.append(x1)
            Y1.append(y1)

        if x2 >= min_x and x2 <= max_x:
            X2.append(x2)
            Y2.append(y2)

    data_1 = {
        "x": X1,
        "y": Y1
    }

    data_2 = {
        "x": X2,
        "y": Y2
    }

    result.append(pd.DataFrame(data_1))
    result.append(pd.DataFrame(data_2))

    return result


if __name__ == "__main__":
    print("Versions")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Pandas : {pd.__version__}")

    # Parse command args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save genereated dataset to file", action="store_true")
    parser.add_argument("-n", "--noise", help="generate dataset with outliers", action="store_true")
    parser.add_argument("-b", "--border", help="show borders of all classes", action="store_true")
    parser.add_argument("-c", "--count", help="dataset points count", type=int)
    parser.add_argument("-t", "--type", help="generator function type", type=int)

    args = parser.parse_args()

    if args.count:
        EXAMPLES_COUNT = args.count

    if args.type:
        FUNC_TYPE = args.type

    # Generate data
    is_noised = True if args.noise else False
    df_generated_dataset = generate(
        EXAMPLES_COUNT, 
        func_type=FUNC_TYPE,
        noise=is_noised,
        noise_alpha=NOISE_ALPHA,
        noise_level=NOISE_LEVEL
    )

    show_info(df_generated_dataset)

    borders = None
    if args.border:
        borders = get_borders(func_type=FUNC_TYPE)

    visualize(data=df_generated_dataset, borders=borders)

    # Save data to file
    if args.save:
        file_format = "csv"

        if borders is not None:
            borders_count = len(borders)
            for i in range(borders_count):
                border_filename = f"multilabel_classification_v{FUNC_TYPE}_border_{i}.{file_format}"
                borders[i].to_csv(border_filename, index=False)
                print(f"Class border #{i} saved to file : {border_filename}")

        dataset_filename = f"multilabel_classification_v{FUNC_TYPE}_{EXAMPLES_COUNT}"
        if is_noised:
            dataset_filename += "_with_noise"
        dataset_filename += f".{file_format}"

        df_generated_dataset.to_csv(dataset_filename, index=False)
        print(f"Dataset saved to file : {dataset_filename}")

    print("Success")
