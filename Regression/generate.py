import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def generate_v1() -> pd.DataFrame:
    min_f1, max_f1 = 0.0, 15.0
    min_f2, max_f2 = 0.0, 15.0

    F1 = min_f1 + np.random.rand(EXAMPLES_COUNT) * max_f1
    F2 = min_f2 + np.random.rand(EXAMPLES_COUNT) * max_f2

    Y = np.sin(F1) + np.cos(F2)

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


    # Generate data
    df_generated_dataset = generate_v1()
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
    dataset_filename = f"generated_data_v1_{EXAMPLES_COUNT}.csv"
    df_generated_dataset.to_csv(dataset_filename, index=False)
    print(f"Dataset saved to file : {dataset_filename}")


    print("Success")