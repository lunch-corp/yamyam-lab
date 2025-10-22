import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constant.metric.metric import Metric

sns.set_style("darkgrid")


def plot_metric_at_k(
    metric: Dict[int, Dict[str, Any]],
    tr_loss: List[float],
    parent_save_path: str,
    top_k_values_for_pred: List[int],
    top_k_values_for_candidate: List[int],
) -> None:
    """
    Draw metrics line plot @k at each epoch after training.
    For direct recommendation, @3,7,10,20 diners will be used.
    For candidate generation, @100,300,500 diners will be used.
    Number of items to used in metrics are different depending on the purpose.

    Args:
        metric (Dict[int, Dict[str, Any]]): metric object after training.
        tr_loss (List[float]): train loss value in each epoch.
        parent_save_path (str): parent save path which will be joined with metric name.
        top_k_values_for_pred (List[int]): List of top k values for prediction metric.
        top_k_values_for_candidate (List[int]): List of top k values for candidate generation metric.
    """
    pred_metrics = [
        Metric.MAP.value,
        Metric.NDCG.value,
    ]
    candidate_metrics = [
        Metric.RECALL.value,
    ]

    if len(top_k_values_for_pred) >= 1:
        for metric_name in pred_metrics:
            pred_metrics_df = pd.DataFrame()
            for k in top_k_values_for_pred:
                epochs = len(metric[k][metric_name])
                tmp = pd.DataFrame(
                    {
                        "metric": metric_name,
                        "@k": f"@{k}",
                        "value": metric[k][metric_name],
                        "epochs": [i for i in range(epochs)],
                    }
                )
                pred_metrics_df = pd.concat([pred_metrics_df, tmp])
            plot_metric(
                df=pred_metrics_df,
                metric_name=metric_name,
                save_path=os.path.join(parent_save_path, f"{metric_name}.png"),
            )

    if len(top_k_values_for_candidate) >= 1:
        for metric_name in candidate_metrics:
            candidate_metrics_df = pd.DataFrame()
            for k in top_k_values_for_candidate:
                epochs = len(metric[k][metric_name])
                tmp = pd.DataFrame(
                    {
                        "metric": metric_name,
                        "@k": f"@{k}",
                        "value": metric[k][metric_name],
                        "epochs": [i for i in range(epochs)],
                    }
                )
                candidate_metrics_df = pd.concat([candidate_metrics_df, tmp])
            plot_metric(
                df=candidate_metrics_df,
                metric_name=metric_name,
                save_path=os.path.join(parent_save_path, f"{metric_name}.png"),
            )

    tr_loss_df = pd.DataFrame(
        {
            "metric": "tr_loss",
            "value": tr_loss,
            "epochs": [i for i in range(len(tr_loss))],
        }
    )
    plot_metric(
        df=tr_loss_df,
        metric_name="tr_loss",
        save_path=os.path.join(parent_save_path, "tr_loss.png"),
        hue=False,
    )


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    save_path: str,
    hue=True,
) -> None:
    """
    Draw line plot given dataframe.

    Args:
        df (pd.DataFrame): dataframe which contains information about metric.
        metric_name (str): metric name to be plotted.
        save_path (str): path to save line plot.
        hue (bool, optional): whether to hue the line plot. Defaults to True.
    """
    if hue is True:
        sns.lineplot(x="epochs", y="value", data=df, hue="@k", marker="o")
        title = f"{metric_name} at @k with every epoch"
    else:
        sns.lineplot(x="epochs", y="value", data=df, marker="o")
        title = f"{metric_name} with every epoch"
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()
