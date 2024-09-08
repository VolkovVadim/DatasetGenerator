import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


EXAMPLES_COUNT     = 5000


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


def generate(func_type: int = 1) -> pd.DataFrame:
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

    data = {
        "feature_1": F1,
        "feature_2": F2,
        "value": Y
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

    args = parser.parse_args()

    if args.count:
        EXAMPLES_COUNT = args.count


    # Generate data
    func_type = 3
    df_generated_dataset = generate(func_type)
    show_info(df_generated_dataset)


    # Visualize data
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("feature_1")
    ax.set_ylabel("feature_2")
    ax.set_zlabel("value")

    ax.xaxis.label.set_color("blue")
    ax.yaxis.label.set_color("blue")
    ax.zaxis.label.set_color("red")

    ax.scatter(
        df_generated_dataset['feature_1'],
        df_generated_dataset['feature_2'],
        df_generated_dataset['value'],
        c = df_generated_dataset['value'],  # values for cmap
        s = 2,                              # marker size
        cmap='viridis'
    )

    plt.show()


    # Save data to file
    if args.save:
        dataset_filename = f"regression_v{func_type}_{EXAMPLES_COUNT}.csv"
        df_generated_dataset.to_csv(dataset_filename, index=False)
        print(f"Dataset saved to file : {dataset_filename}")


    print("Success")