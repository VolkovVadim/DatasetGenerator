import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


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


def visualize(data: pd.DataFrame, borders: pd.DataFrame = None) -> None:
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
    plt.title(f'Binary classification ({points_count} points)')

    if borders is not None:
        plt.plot(
            borders.x1,
            borders.y1,
            linestyle='--',
            linewidth=2,
            c='midnightblue',
        )

    color = ['red' if label == 0 else 'blue' for label in data.class_label]

    point_size = 3 if data.shape[0] < 50000 else 1

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
        Y = [1 if F1[i] - F2[i] > 0 else 0 for i in range(count)]

    if func_type == 2:
        Y = [1 if F1[i] - 10.0 * np.sin(0.25 * F2[i]) > 0 else 0 for i in range(count)]

    if func_type == 3:
        Y = [1 if F1[i] - (F2[i] + 5.5 * np.sin(0.5 * F2[i])) > 0 else 0 for i in range(count)]

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


def get_borders(func_type: int = 1) -> pd.DataFrame:
    if func_type < 1 or func_type > 3:
        func_type = 1

    min_x, max_x = -25.0, 25.0

    Y = np.arange(min_x, max_x, 0.25)

    if func_type == 1:
        X = [Y[i] for i in range(len(Y))]

    if func_type == 2:
        X = [10.0 * np.sin(0.25 * Y[i]) for i in range(len(Y))]

    if func_type == 3:
        X = [Y[i] + 5.5 * np.sin(0.5 * Y[i]) for i in range(len(Y))]

    data = {
        "x1": X,
        "y1": Y
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Versions")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Pandas : {pd.__version__}")

    # Parse command args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save genereated dataset to file", action="store_true")
    parser.add_argument("-n", "--noise", help="generate dataset with outliers", action="store_true")
    parser.add_argument("-b", "--border", help="show class border", action="store_true")
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

    df_borders = None
    if args.border:
        df_borders = get_borders(func_type=FUNC_TYPE)

    show_info(df_generated_dataset)
    visualize(data=df_generated_dataset, borders=df_borders)

    # Save data to file
    if args.save:
        file_format = "csv"

        if df_borders is not None:
            border_filename = f"binary_classification_v{FUNC_TYPE}_border.{file_format}"
            df_borders.to_csv(border_filename, index=False)
            print(f"Class border data saved to file : {border_filename}")

        dataset_filename = f"binary_classification_v{FUNC_TYPE}_{EXAMPLES_COUNT}"
        if is_noised:
            dataset_filename += "_with_noise"
        dataset_filename += f".{file_format}"

        df_generated_dataset.to_csv(dataset_filename, index=False)
        print(f"Dataset saved to file : {dataset_filename}")

    print("Success")


