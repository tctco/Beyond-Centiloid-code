import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import warnings

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_context("paper")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_linear_corr(
    pred,
    gt,
    xlabel: str,
    xlim=None,
    ylabel="Ground Truth",
    plot_ref=False,
    figsize=(3, 3),
):
    x_with_constant = sm.add_constant(pred)
    model = sm.OLS(gt, x_with_constant).fit()

    params = model.params
    r_squared = model.rsquared
    if params[1] < 0:
        line_eq = f"y = {params[1]:.2f}x - {abs(params[0]):.2f}"
    else:
        line_eq = f"y = {params[1]:.2f}x + {params[0]:.2f}"
    r2_eq = f"R2 = {r_squared:.4f}"

    data = pd.DataFrame({xlabel: pred, ylabel: gt})
    sns.despine()
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(data=data, x=xlabel, y=ylabel, ax=ax, scatter=False)
    # g = sns.lmplot(
    #     x=xlabel,
    #     y="Ground Truth",
    #     data=data,
    #     fit_reg=False,
    #     height=3,
    #     markers="o",
    #     legend=True,
    #     legend_out=False,
    #     palette=[sns.color_palette()[0], sns.color_palette()[0]],
    # )
    # g.ax.legend_.set_title(None)
    # plot a line of best fit
    if xlim is not None:
        y = [params[1] * xlim[0] + params[0], params[1] * xlim[1] + params[0]]
        plt.plot(xlim, y, color=sns.color_palette()[0])
    else:
        x = [min(pred), max(pred)]
        y = [params[1] * x[0] + params[0], params[1] * x[1] + params[0]]
        plt.plot(x, y, color=sns.color_palette()[0])
    ax.text(
        0.25,
        0.95 - 0.02,
        line_eq,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.25,
        0.90 - 0.02,
        r2_eq,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    plt.ylabel(ylabel)
    if plot_ref:
        # plot x = y
        plt.plot([min(gt), max(gt)], [min(gt), max(gt)], color="black", linestyle="--")
    # plt.legend(title=None, loc='lower right')
    plt.legend("", frameon=False)
    return fig, ax


def plot_brain_region_heatmap():
    pass


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def register_alpha_colormap(cmap_name):
    ncolors = 256
    color_array = plt.get_cmap(cmap_name)(range(ncolors))

    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors) ** 0.5
    map_object = LinearSegmentedColormap.from_list(
        name=f"{cmap_name}_alpha", colors=color_array
    )
    try:
        plt.colormaps.register(cmap=map_object)
    except ValueError as e:
        warnings.warn(str(e))
    return f"{cmap_name}_alpha"


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(
    fpr,
    tpr,
    auc_value,
    tpr_upper=None,
    tpr_lower=None,
    color="blue",
    plot_random_classifier=True,
    prefix="ROC curve",
    ax=None,
):
    """
    This function plots the ROC curve along with optional confidence intervals for the true positive rate (TPR)
    at each false positive rate (FPR). If confidence intervals are provided, they are plotted as shaded regions
    around the ROC curve.

    Parameters:
    -----------
    fpr : array-like
        False positive rates for the ROC curve.

    tpr : array-like
        True positive rates for the ROC curve.

    auc_value : float
        The area under the ROC curve (AUC) value.

    tpr_upper : array-like, optional (default=None)
        The upper bound of the TPR for the confidence interval.

    tpr_lower : array-like, optional (default=None)
        The lower bound of the TPR for the confidence interval.

    ax : matplotlib.axes.Axes, optional (default=None)
        The axes to plot on. If None, a new figure and axes are created.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    # If no Axes are provided, create a new figure and Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    # Plot the main ROC curve
    label = f"{prefix} (AUC = {auc_value:.2f})"
    # If confidence intervals are provided, plot the shaded region
    if tpr_upper is not None and tpr_lower is not None:
        auc_upper = auc(fpr, tpr_upper)
        auc_lower = auc(fpr, tpr_lower)
        ax.fill_between(fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)
        label += "\n" + f"95% CI: {auc_lower:.2f}-{auc_upper:.2f}"
    ax.plot(fpr, tpr, color=color, label=label)

    if plot_random_classifier:
        # Plot diagonal line for random classifier (AUC = 0.5)
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            linestyle="--",
            # lw=2,
            label="Random (AUC = 0.5)",
        )

    # Set plot labels and title
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")

    # Display the legend
    ax.legend(loc="lower right")

    # Show the plot
    plt.tight_layout()

    return ax


import numpy as np
from sklearn.metrics import roc_curve
from sklearn.utils import resample


def calculate_roc_with_ci(
    label, pred_prob, confidence_interval=95, n_bootstrap=1000, random_seed=42
):
    """
    This function computes the ROC curve, AUC, and estimates the confidence intervals (CI) for the TPR at each FPR
    using bootstrap resampling. The function returns the FPR values, the mean TPR, and the upper and lower bounds
    of the TPR at each FPR, corresponding to the specified confidence interval.

    Parameters:
    -----------
    label : array-like, shape (n_samples,)
        True binary labels (0 or 1) for the classification task.

    pred_prob : array-like, shape (n_samples,)
        The predicted probabilities for the positive class (class 1).

    confidence_interval : float, optional (default=95)
        The confidence level for the TPR confidence intervals (e.g., 95 means 95% CI).

    n_bootstrap : int, optional (default=1000)
        The number of bootstrap resamples to perform.

    random_seed : int, optional (default=42)
        The random seed for reproducibility.

    Returns:
    --------
    fpr_values : array-like
        The false positive rate values for the ROC curve.

    mean_tpr : array-like
        The mean true positive rate (TPR) at each FPR across bootstrap samples.

    tpr_upper : array-like
        The upper bound (e.g., 97.5 percentile) of the TPR at each FPR for the confidence interval.

    tpr_lower : array-like
        The lower bound (e.g., 2.5 percentile) of the TPR at each FPR for the confidence interval.
    """

    np.random.seed(random_seed)

    # Calculate the ROC curve for the original data
    fpr, tpr, _ = roc_curve(label, pred_prob)

    # Initialize a list to store the ROC curves from bootstrap resampling
    bootstrap_roc = []

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Bootstrap resample (with replacement)
        indices = resample(np.arange(len(label)))
        y_resampled = label[indices]
        pred_prob_resampled = pred_prob[indices]

        # Calculate the ROC curve for the resampled data
        fpr_resampled, tpr_resampled, _ = roc_curve(y_resampled, pred_prob_resampled)
        bootstrap_roc.append((fpr_resampled, tpr_resampled))

    # Compute the unique FPR values across all bootstrap samples
    fpr_values = np.unique(np.concatenate([roc[0] for roc in bootstrap_roc]))

    # Initialize arrays for storing the mean TPR and the upper/lower bounds of TPR
    mean_tpr = np.zeros_like(fpr_values)
    tpr_upper = np.zeros_like(fpr_values)
    tpr_lower = np.zeros_like(fpr_values)

    # For each unique FPR, compute the corresponding TPR values and confidence intervals
    for i, fpr_val in enumerate(fpr_values):
        # Interpolate TPR values at the current FPR across all bootstrap samples
        tprs_at_fpr = [np.interp(fpr_val, roc[0], roc[1]) for roc in bootstrap_roc]

        # Calculate the mean TPR
        mean_tpr[i] = np.mean(tprs_at_fpr)

        # Calculate the lower and upper bounds for the confidence interval
        lower_percentile = (100 - confidence_interval) / 2
        upper_percentile = 100 - lower_percentile
        tpr_lower[i] = np.percentile(tprs_at_fpr, lower_percentile)
        tpr_upper[i] = np.percentile(tprs_at_fpr, upper_percentile)
    # add 0, 0 and 1, 1
    fpr_values = np.concatenate([[0], fpr_values, [1]])
    mean_tpr = np.concatenate([[0], mean_tpr, [1]])
    tpr_upper = np.concatenate([[0], tpr_upper, [1]])
    tpr_lower = np.concatenate([[0], tpr_lower, [1]])
    auc_lower = np.trapz(tpr_lower, fpr_values)
    auc_upper = np.trapz(tpr_upper, fpr_values)
    return fpr_values, mean_tpr, tpr_upper, tpr_lower, auc_lower, auc_upper


# Example usage:
# fpr_values, mean_tpr, tpr_upper, tpr_lower = calculate_roc_with_ci(y_gt, pred_prob, confidence_interval=95)


def evaluate_binary_classification(
    ground_truth,
    pred_prob,
    threshold=0.5,
    labels=["Class 0", "Class 1"],
    color="blue",
    roc_ax=None,
    dataset_name=None,
    plot_random_classifier=False,
    n_bootstrap=1000,
):
    """
    Evaluate binary classification performance based on ground_truth and predicted probabilities.

    Args:
    - ground_truth (array-like): True labels (0 or 1).
    - pred_prob (array-like): Predicted probabilities of the positive class.
    - threshold (float): The threshold for converting probabilities to binary predictions. Default is 0.5.

    Returns:
    - dict: A dictionary containing evaluation metrics.
    """

    # Convert probabilities to binary predictions
    pred_labels = (pred_prob >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, pred_labels)
    precision = precision_score(ground_truth, pred_labels)
    recall = recall_score(ground_truth, pred_labels)
    auc = roc_auc_score(ground_truth, pred_prob)
    f1 = f1_score(ground_truth, pred_labels)
    fpr_values, mean_tpr, tpr_upper, tpr_lower, *_ = calculate_roc_with_ci(
        ground_truth, pred_prob, n_bootstrap=n_bootstrap
    )

    # Confusion Matrix
    cm = confusion_matrix(ground_truth, pred_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Plot ROC Curve
    if roc_ax is None:
        roc_fig, roc_ax = plt.subplots(figsize=(3.5, 3.5))
    else:
        roc_fig = None
    sns.despine(ax=roc_ax)
    roc_ax = plot_roc(
        fpr_values,
        mean_tpr,
        auc,
        tpr_upper,
        tpr_lower,
        color=color,
        ax=roc_ax,
        plot_random_classifier=plot_random_classifier,
        prefix="ROC curve" if dataset_name is None else dataset_name,
    )

    # Plot Confusion Matrix
    cm_fig, cm_ax = plt.subplots(figsize=(2, 2))
    sns.despine(ax=cm_ax, left=True, bottom=True)
    # cm_display.plot(cmap=plt.cm.Purples, ax=cm_ax, colorbar=False)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", ax=cm_ax, square=True, cbar=False
    )
    # set ticks
    cm_ax.set_xticklabels(labels, rotation=0)
    cm_ax.set_yticklabels(labels, rotation=0)
    cm_ax.set_xlabel("Predicted Label")
    cm_ax.set_ylabel("True Label")

    # Return metrics as a dictionary
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "ROC Curve": [roc_fig, roc_ax],
        "Confusion Matrix": [cm_fig, cm_ax],
    }

    return metrics


from math import pi as PI


def radar(df, hue, ylim=None):

    # Set data
    # df = pd.DataFrame({
    # 'group': ['A','B','C','D'],
    # 'var1': [38, 1.5, 30, 4],
    # 'var2': [29, 10, 9, 34],
    # 'var3': [8, 39, 23, 24],
    # 'var4': [7, 31, 33, 14],
    # 'var5': [28, 15, 32, 14]
    # })

    # ------- PART 1: Create background

    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * PI for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # ax = plt.subplot(111, polar=True)
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))

    # If you want the first axis to be on top:
    ax.set_theta_offset(PI / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    # plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    # plt.ylim(0, 40)
    if ylim is not None:
        plt.ylim(0, ylim)
        # set 4 ticks
        ax.set_yticks(np.linspace(0, ylim, 5))

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    group_name = df.loc[0, hue]
    values = df.loc[0].drop(hue).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle="solid", label=group_name)
    ax.fill(angles, values, alpha=0.1)

    # Ind2
    group_name = df.loc[1, hue]
    values = df.loc[1].drop(hue).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle="solid", label=group_name)
    ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # Show the graph
    return fig, ax
