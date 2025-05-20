import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_diff(file_path='swat_result.csv', sensors=(11, 13, 23), start=17200, end=17380, save_fig=0,
              fig_path='diff_plot_11_13_23_17200_17380.png'):
    df = pd.read_csv(file_path)
    if 'label' not in df.columns:
        raise ValueError('Column "label" not found')

    indices = range(start, end + 1)
    if end >= len(df):
        raise IndexError('end index out of range')

    time_slice = df.iloc[indices]
    plt.figure(figsize=(12, 6))
    for sensor in sensors:
        pred_col = f'sensor{sensor}_pred'
        true_col = f'sensor{sensor}_true'
        if pred_col not in df.columns or true_col not in df.columns:
            raise ValueError(f'Columns for sensor {sensor} not found')

        diff = abs(time_slice[pred_col] - time_slice[true_col])#/abs(time_slice[true_col])
        plt.plot(indices, diff, label=f'Sensor {sensor}')


    for idx, lbl in zip(indices, time_slice['label']):
        if lbl == 1:
            plt.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.1)  # Lightened shading

    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.title('Prediction Error for Sensors')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_fig:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    plot_diff()