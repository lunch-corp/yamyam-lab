from typing import Dict, Any, List
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from constant.evaluation.recommend import TOP_K_VALUES_FOR_PRED, TOP_K_VALUES_FOR_CANDIDATE
from constant.metric.metric import Metric, NearCandidateMetric


def plot_metric_at_k(
        metric: Dict[int, Dict[str, Any]],
        tr_loss: List[float],
        parent_save_path: str,
) -> None:
    pred_metrics = [
        Metric.MAP.value,
        Metric.NDCG.value,
        NearCandidateMetric.RANKED_PREC.value,
    ]
    candidate_metrics = [
        NearCandidateMetric.NEAR_RECALL.value,
        Metric.RECALL.value,
    ]

    for metric_name in pred_metrics:
        pred_metrics_df = pd.DataFrame()
        for k in TOP_K_VALUES_FOR_PRED:
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

    for metric_name in candidate_metrics:
        candidate_metrics_df = pd.DataFrame()
        for k in TOP_K_VALUES_FOR_CANDIDATE:
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
    )

def plot_metric(
        df: pd.DataFrame,
        metric_name: str,
        save_path: str,
        hue=True,
) -> None:
    if hue is True:
        sns.lineplot(x="epochs", y="value", data=df, hue="@k", marker="o")
        title = f"{metric_name} at @k with every epoch"
    else:
        sns.lineplot(x="epochs", y="value", data=df, marker="o")
        title = f"{metric_name}with every epoch"
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()