# ==============================================================================
# plotting.py
# ==============================================================================
# Functions for visualizing model performance.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf


def plot_pacf_solar(df, lags=48, save_dir="figures"):
    """
    Plot the PACF for the solar series ('solar_photovoltaic_mw').

    Utility: Visualizes partial autocorrelation to understand lags in solar data.

    Arguments:
    - df: DataFrame, dataset.
    - lags: int, number of lags to plot.
    - save_dir: str, directory to save figure.

    Returns:
    - None (plots and saves figure).
    """
    if "solar_photovoltaic_mw" not in df.columns:
        print("The column 'solar_photovoltaic_mw' is absent from the DataFrame.")
        return
    
    series = df["solar_photovoltaic_mw"].dropna()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_pacf(series, lags=lags, method="ywm", ax=ax)
    
    ax.set_title(f"PACF - Solar Photovoltaic (lags={lags})", fontsize=14, weight="bold")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Partial Autocorrelation")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    filename = f"{save_dir}/pacf_solar_photovoltaic.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {filename}")
    
    plt.show()

def plot_correlations(corrs, by='output', save_dir='figures'):
    """
    Plot Spearman correlation matrices.

    Utility: Visualizes average average correlations between outputs or horizons for each model.

    Arguments:
    - corrs: dict {model: DataFrame}, correlation matrices.
    - by: str, 'output' or 'horizon'.
    - save_dir: str, save directory.

    Returns:
    - None (plots and saves).
    """
    if not corrs:
        print("No correlations to plot.")
        return
    
    for model, corr_mat in corrs.items():
        # CHANGEMENT: Taille plus grande for 'horizon' (15x12) pour éviter cases trop petites
        if by == 'horizon':
            fig, ax = plt.subplots(figsize=(15, 12))
            annot_size = 6  # Taille des annotations réduite pour lisibilité
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            annot_size = 10  # Taille normale for 'output'
        
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax,
                    annot_kws={"size": annot_size})  # CHANGEMENT: annot_kws pour ajuster taille des nombres
        
        title = f"Correlation Spearman {by.capitalize()} for {model} (Average)"
        ax.set_title(title)
        plt.tight_layout()
        
        filename = f"{save_dir}/correlation_spearman_{by}_{model}.png"
        plt.savefig(filename, dpi=300)
        print(f"Figure saved: {filename}")
        plt.show()

def plot_metrics_subplots(metrics_df, models_to_plot=None, show_legend=True):
    """
    Plots RMSE, R2, MAE, and MBE in a 2x2 subplot figure.

    Utility: Compares model performance across horizons.

    Arguments:
    - metrics_df: DataFrame, metrics data.
    - models_to_plot: list of str, models to include (optional).
    - show_legend: bool, show legend.

    Returns:
    - None (plots).
    """
    print(">>> Generating plots...")
    print("    - Generating Plot 1: Key Metrics Subplots (RMSE, R2, MAE, MBE)...")
    
    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    models = sorted(metrics_df["model"].unique())
    markers = ['o', 'x', 's', '^', 'd', 'v']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Model Performance Comparison Across Horizons", fontsize=16)
    
    metric_names = ["RMSE", "R2", "MAE", "MBE"]
    
    for ax, metric in zip(axs.flat, metric_names):
        for model in models:
            # Calculate the mean metric value across all outputs for each horizon
            data = (metrics_df[metrics_df["model"] == model]
                    .groupby("horizon", observed=True)[metric]
                    .mean())
            ax.plot(data.index, data.values, marker=marker_map[model], linestyle='-', label=model)
        
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if show_legend:
        ax.legend(
            fontsize=14,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=True
        )
    
    # Common X-axis label
    axs[1, 0].set_xlabel("Forecast Horizon (hours)")
    axs[1, 1].set_xlabel("Forecast Horizon (hours)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_normalized_metrics_subplots(metrics_df, models_to_plot=None, show_legend=True):
    """
    Plots nRMSE, R2, nMAE, and nMBE in a 2x2 subplot figure.

    Utility: Compares normalized model performance across horizons.

    Arguments: (same as plot_metrics_subplots)

    Returns:
    - None (plots and saves).
    """
    print("    - Generating Plot 2: Normalized Key Metrics Subplots (nRMSE, R2, nMAE, nMBE)...")
    
    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    models = sorted(metrics_df["model"].unique())
    markers = ['o', 'x', 's', '^', 'd', 'v']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Model Performance Comparison Across Horizons (Normalized)", fontsize=16)
    
    metric_names = ["nRMSE", "R2", "nMAE", "nMBE"]
    
    for ax, metric in zip(axs.flat, metric_names):
        for model in models:
            # Calculate the mean metric value across all outputs for each horizon
            data = (metrics_df[metrics_df["model"] == model]
                    .groupby("horizon", observed=True)[metric]
                    .mean())
            ax.plot(data.index, data.values, marker=marker_map[model], linestyle='-', label=model)
        
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if show_legend:
        ax.legend(
            fontsize=14,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=True
        )
    
    # Common X-axis label
    axs[1, 0].set_xlabel("Forecast Horizon (hours)")
    axs[1, 1].set_xlabel("Forecast Horizon (hours)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Savefig
    filename = f"figures/model_comparison_normalized_metrics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")
    
    plt.show()

def plot_all_metrics_by_energy_variable(metrics_df, models_to_plot=None):
    """
    Plots nRMSE, nMAE, nMBE, and R2 for each output variable on separate large plots.

    Utility: Visualizes metrics by variable and model.

    Arguments:
    - metrics_df: DataFrame, metrics.
    - models_to_plot: list, models to include.

    Returns:
    - None (plots and saves).
    """
    print("    - Generating Plot 3: Metrics by Energy Variable (Combined View)...")

    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    outputs = sorted(metrics_df["output"].unique())
    models = sorted(metrics_df["model"].unique())
    
    # Define consistent styling for models
    styles = ['-', '--', ':', '-.']
    markers = ['o', 'x', 's', '^', 'd', 'v', '<', '>']
    linestyle_by_model = {model: styles[i % len(styles)] for i, model in enumerate(models)}
    marker_by_model = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    palette = sns.color_palette("tab10", len(outputs))
    color_map = {out: palette[i] for i, out in enumerate(outputs)}


    # List of metrics to plot
    metrics_to_plot = ["nRMSE", "nMAE", "nMBE", "R2"]

    for metric in metrics_to_plot:
        print(f"      - Plotting {metric}...")
        fig, ax = plt.subplots(figsize=(20, 14), constrained_layout=True)  # figsize plus grand + constrained_layout

        for out in outputs:
            for model in models:
                data = (metrics_df[(metrics_df["output"] == out) & (metrics_df["model"] == model)]
                        .sort_values("horizon"))
                if data.empty:
                    continue
                
                ax.plot(
                    data["horizon"], data[metric],
                    label=f"{out} ({model})",
                    linestyle=linestyle_by_model[model],
                    marker=marker_by_model[model],
                    color=color_map[out],
                    alpha=0.8
                )

        ax.set_xlabel("Forecast Horizon (hours)", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f"{metric} vs. Horizon by Energy Variable and Model", fontsize=18, weight="bold")
        ax.grid(True, which='both', linestyle='--', linewidth=0.6)

        # Custom legends
        output_handles = [Line2D([0], [0], color=color_map[o], lw=2, label=o) for o in outputs]
        model_handles = [Line2D([0], [0], color='gray', lw=2, linestyle=linestyle_by_model[m], 
                                marker=marker_by_model[m], label=m) for m in models]

        # Légende des variables en haut
        legend1 = ax.legend(handles=output_handles, title="Energy Variable", fontsize=15, title_fontsize=13,
                            loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(outputs)))
        ax.add_artist(legend1)

        # Légende des modèles en bas
        ax.legend(handles=model_handles, title="Model", fontsize=15, title_fontsize=16,
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(6, len(models)))

        # Sauvegarde
        filename = f"figures/model_comparison_{metric.lower()}_by_variable.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Figure saved: {filename}")
        
        plt.show()


def plot_metrics_by_variable_subplots(metrics_df, models_to_plot=None, show_legend=True):
    """
    Creates a grid of subplots for each metric. Each subplot shows the model comparison for a single energy variable.

    Utility: Subplot view for metrics by variable.

    Arguments: (similar to plot_all_metrics_by_energy_variable)

    Returns:
    - None (plots and saves).
    """
    print("    - Generating Plot 4: Metrics by Energy Variable (Subplot View)...")

    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    outputs = sorted(metrics_df["output"].unique())
    models = sorted(metrics_df["model"].unique())
    metrics_to_plot = ["nRMSE", "R2", "nMAE", "nMBE"]
    
    markers = ['o', 'x', 's', '^', 'd', 'v']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Determine grid size
    n_outputs = len(outputs)
    n_cols = 2
    n_rows = (n_outputs + n_cols - 1) // n_cols

    for metric in metrics_to_plot:
        print(f"      - Plotting {metric}...")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
        axes = axes.flatten()
        
        fig.suptitle(f"{metric} Comparison Across Models by Energy Variable", fontsize=16, y=1.02)

        for i, var in enumerate(outputs):
            ax = axes[i]
            for model in models:
                subset = metrics_df[(metrics_df['output'] == var) & (metrics_df['model'] == model)]
                if not subset.empty:
                    ax.plot(subset['horizon'], subset[metric], marker=marker_map[model], linestyle='-', label=model)
            
            ax.set_title(var, fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if show_legend:
            ax.legend(
                fontsize=12,
                loc="center left",      # Position sur le côté
                bbox_to_anchor=(1, 0.5) # Décale à droite
            )

        # Add common X-axis labels to the bottom plots
        for i in range(n_rows * n_cols - n_cols, n_rows * n_cols):
             if i < n_outputs:
                axes[i].set_xlabel("Forecast Horizon (hours)")

        # Hide any unused subplots
        for i in range(n_outputs, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Savefig
        filename = f"figures/{metric.lower()}_by_variable_subplots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")

        plt.show()

    print(">>> Plot generation complete.\n")

def plot_reconciliation_improvement(metrics_df):
    """
    Plots the % improvement in average nRMSE due to reconciliation for each model type across horizons.

    Utility: Shows the difference and utility of reconciliation by quantifying % reduction in nRMSE.

    Arguments:
    - metrics_df: DataFrame, final metrics.

    Returns:
    - None (plots and saves).
    """
    print("    - Generating Plot 5: Reconciliation Improvement (% in nRMSE)...")
    model_types = ['SISO', 'MIMO', 'MIMO-MH']
    fig, ax = plt.subplots(figsize=(12, 8))
    has_data = False
    for base in model_types:
        rec = base + '-REC'
        base_data = metrics_df[metrics_df['model'] == base].groupby('horizon')['nRMSE'].mean()
        rec_data = metrics_df[metrics_df['model'] == rec].groupby('horizon')['nRMSE'].mean()
        if not base_data.empty and not rec_data.empty:
            has_data = True
            horizons = sorted(set(base_data.index) & set(rec_data.index))
            improvement = []
            for h in horizons:
                b_val = base_data.get(h, np.nan)
                r_val = rec_data.get(h, np.nan)
                if not np.isnan(b_val) and not np.isnan(r_val) and b_val != 0:
                    imp = (b_val - r_val) / b_val * 100
                    improvement.append(imp)
                else:
                    improvement.append(np.nan)
            ax.plot(horizons, improvement, marker='o', label=f'{base} Improvement %')
    if has_data:
        ax.set_title('Reconciliation Utility: % Improvement in Average nRMSE')
        ax.set_xlabel('Forecast Horizon (hours)')
        ax.set_ylabel('% Improvement (lower nRMSE)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        filename = "figures/reconciliation_improvement.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
        plt.show()
    else:
        print("    - No reconciliation pairs found for improvement plot.")

def plot_model_times(model_times):
    """
    Plots a bar chart of execution times per model.

    Utility: Visualizes the time taken by each model.

    Arguments:
    - model_times: dict {model_name: time_seconds}

    Returns:
    - None (plots and saves).
    """
    if not model_times:
        print("    - No model times to plot.")
        return
    print("    - Generating Plot 6: Model Execution Times...")
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(model_times.keys())
    times = list(model_times.values())
    ax.bar(models, times)
    ax.set_title('Execution Time per Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    filename = "figures/model_execution_times.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")
    plt.show()