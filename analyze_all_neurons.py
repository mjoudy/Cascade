import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cascade2p import utils
import kernel_est_funcs as kef
import remove_outliers as ro

# Parameters
UPSAMPLE_RATE = 1000
SMOOTH_WIN = 301
CUT_WIN = 100

def process_neuron(times, dff, spikes):
    # 1. Upsample
    ups_signal, ups_spikes = kef.upsample(times, dff, spikes, UPSAMPLE_RATE)
    # 2. Smoothing
    ups_smooth_signal, ups_smooth_deriv = kef.smoothed_signals(ups_signal, SMOOTH_WIN)
    # 3. Spike cutting
    cut_signal, cut_deriv = kef.cut_spikes(ups_spikes, ups_smooth_signal, ups_smooth_deriv, win_len=CUT_WIN)
    # 4. Kernel fitting (tau estimation)
    try:
        tau_inv = kef.pure_fit(cut_signal, cut_deriv)
        tau = -1/tau_inv if tau_inv != 0 else np.nan
    except Exception as e:
        tau = np.nan
    # 5. Cumulative spikes
    n_spikes = len(spikes)
    # 6. Upsampling quality (correlation between upsampled spikes and reconstructed signal)
    try:
        reconst = ups_smooth_deriv + (-tau)*ups_smooth_signal
        # Create binary spike train
        spikes_train = np.zeros(np.shape(ups_smooth_signal))
        ups_spikes_int = ups_spikes.astype(int)
        ups_spikes_int = ups_spikes_int[ups_spikes_int < len(spikes_train)]
        spikes_train[ups_spikes_int] = 1
        corr = np.corrcoef(spikes_train, reconst)[0,1]
    except Exception as e:
        corr = np.nan
    return tau, n_spikes, corr

def main():
    # Load all ground truth datasets
    datasets = utils.load_all_ground_truth('Ground_truth')
    results = []
    for dataset_name, recordings in datasets.items():
        print(f"Processing dataset: {dataset_name} ({len(recordings)} recordings)")
        for i, rec in enumerate(recordings):
            dff = rec['dff']
            times = rec['t']
            spikes = rec['spikes']
            tau, n_spikes, corr = process_neuron(times, dff, spikes)
            results.append({
                'dataset': dataset_name,
                'neuron': i,
                'tau': tau,
                'n_spikes': n_spikes,
                'corr': corr
            })
    # Convert to structured array
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('neuron_analysis_results.csv', index=False)
    print("Results saved to neuron_analysis_results.csv")
    # Plotting
    plt.figure(figsize=(12,6))
    sns.boxplot(x='dataset', y='tau', data=df)
    plt.xticks(rotation=90)
    plt.title('Distribution of Time Constant (tau) per Dataset')
    plt.tight_layout()
    plt.savefig('tau_per_dataset.png')
    plt.close()

    plt.figure(figsize=(12,6))
    sns.boxplot(x='dataset', y='n_spikes', data=df)
    plt.xticks(rotation=90)
    plt.title('Distribution of Cumulative Spikes per Dataset')
    plt.tight_layout()
    plt.savefig('spikes_per_dataset.png')
    plt.close()

    plt.figure(figsize=(12,6))
    sns.boxplot(x='dataset', y='corr', data=df)
    plt.xticks(rotation=90)
    plt.title('Correlation (Upsampled Spikes vs. Reconstructed Signal) per Dataset')
    plt.tight_layout()
    plt.savefig('corr_per_dataset.png')
    plt.close()

    print("Plots saved: tau_per_dataset.png, spikes_per_dataset.png, corr_per_dataset.png")

if __name__ == '__main__':
    main() 