import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


EXAMPLES_COUNT     = 5000       # Points count in dataset
FUNC_TYPE          = 3          # Dataset type
NOISE_ALPHA        = 0.8        # Part of the dataset with outliers
NOISE_LEVEL        = 0.5        # Data noise level


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


def generate(
        count: int, 
        func_type: int = 1, 
        noise: bool = False,
        noise_alpha: float = 0.1,
        noise_level: float = 1.0
) -> pd.DataFrame:
    min_f1, max_f1 = -25.0, 25.0
    min_f2, max_f2 = -25.0, 25.0

    F1 = min_f1 + np.random.rand(EXAMPLES_COUNT) * max_f1
    F2 = min_f2 + np.random.rand(EXAMPLES_COUNT) * max_f2

    if func_type < 1 or func_type > 3:
        func_type = 1

    if func_type == 1:
        Y = np.sin(F1) + np.cos(F2)

    if func_type == 2:
        Y = np.sin(F1) + np.cos(F2) + np.arctan(F2)

    if func_type == 3:
        Y = 10.0 * F1 - 0.05 * (F2 * F2 * F2) + 75.0 * np.sin(F1) - 55.0 * np.cos(F2)

    if noise:
        examples_with_noise = int(float(count) * noise_alpha)
        selected_examples = np.random.randint(count, size=examples_with_noise)
        for i in selected_examples:
            F1[i] += np.random.normal(scale=noise_level)
            F2[i] += np.random.normal(scale=noise_level)

    data = {
        "feature_1": F1,
        "feature_2": F2,
        "value": Y
    }

    return pd.DataFrame(data)


def visualize(data: pd.DataFrame) -> None:
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12, 10), dpi=80)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("feature_1")
    ax.set_ylabel("feature_2")
    ax.set_zlabel("value")

    ax.xaxis.label.set_color("blue")
    ax.yaxis.label.set_color("blue")
    ax.zaxis.label.set_color("red")

    color = data.value
    points_count = data.shape[0]
    point_size = 5 if points_count < 50000 else 1

    ax.text2D(0.05, 0.95, f"Regression dataset with {points_count} examples", transform=ax.transAxes)

    ax.scatter(
        data['feature_1'],
        data['feature_2'],
        data['value'],
        c = color,           # values for cmap
        s = point_size,      # marker size
        cmap='viridis'
    )

    plt.show()


if __name__ == "__main__":
    print("Versions")
    print(f"  NumPy  : {np.__version__}")
    print(f"  Pandas : {pd.__version__}")


    # Parse command args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="save genereated dataset to file", action="store_true")
    parser.add_argument("-n", "--noise", help="generate dataset with outliers", action="store_true")
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


    # Visualize data
    visualize(df_generated_dataset)


    # Save data to file
    if args.save:
        file_format = "csv"
        dataset_filename = f"binary_classification_v{FUNC_TYPE}_{EXAMPLES_COUNT}"
        if is_noised:
            dataset_filename += "_with_noise"
        dataset_filename += f".{file_format}"

        df_generated_dataset.to_csv(dataset_filename, index=False)
        print(f"Dataset saved to file : {dataset_filename}")


    print("Success")