import copy
import traceback

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from loss.custom import svd_loss
from evaluation.metric import ranking_metrics_at_k
from tools.parse_args import parse_args
from tools.logger import setup_logger
from tools.utils import convert_tensor

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


class SVDWithBias(nn.Module):

    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super(SVDWithBias, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.mu = kwargs["mu"]

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        nn.init.xavier_normal_(self.user_bias.weight)
        nn.init.xavier_normal_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        embed_user = self.embed_user(user_idx)  # batch_size * num_factors
        embed_item = self.embed_item(item_idx)  # batch_size * num_factors
        user_bias = self.user_bias(user_idx)  # batch_size * 1
        item_bias = self.item_bias(item_idx)  # batch_size * 1
        output = (
            (embed_user * embed_item).sum(axis=1) + user_bias.squeeze() + item_bias.squeeze() + self.mu
        )  # batch_size * 1
        return output

    def recommend(self, X_train, X_val, top_K=[3, 5, 7, 10, 20], filter_already_liked=True):

        self.map = 0.0
        self.ndcg = 0.0

        train_liked = convert_tensor(X_train, dict)
        val_liked = convert_tensor(X_val, list)
        res = {}
        metric_at_K = {k: {"map": 0, "ndcg": 0, "count": 0} for k in top_K}
        for user in range(self.num_users):
            item_idx = torch.arange(self.num_items)
            user_idx = torch.tensor([user]).repeat(self.num_items)

            # calculate one user's predicted scores for all item_ids
            with torch.no_grad():
                scores = self.forward(user_idx, item_idx)

            # filter item_id in train dataset
            if filter_already_liked:
                user_liked_items = train_liked[user]
                for already_liked_item_id in user_liked_items.keys():
                    scores[already_liked_item_id] = -float("inf")  # not recommend already chosen item_id

            # calculate metric
            val_liked_item_id = np.array(val_liked[user])
            for K in top_K:
                if len(val_liked_item_id) < K:
                    continue
                pred_liked_item_id = torch.topk(scores, k=K).indices.detach().cpu().numpy()
                metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                metric_at_K[K]["map"] += metric["ap"]
                metric_at_K[K]["ndcg"] += metric["ndcg"]
                metric_at_K[K]["count"] += 1

                # store recommendation result when K=20
                if K == 20:
                    res[user] = pred_liked_item_id
        for K in top_K:
            metric_at_K[K]["map"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ndcg"] /= metric_at_K[K]["count"]
        self.metric_at_K = metric_at_K
        return res


if __name__ == "__main__":
    from preprocess.preprocess import train_test_split_stratify, prepare_torch_dataloader

    args = parse_args()
    logger = setup_logger(args.log_path)

    try:
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"regularization: {args.regularization}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"number of factors for user / item embedding: {args.num_factors}")
        logger.info(f"test ratio: {args.test_ratio}")
        logger.info(f"patience for watching validation loss: {args.patience}")
        data = train_test_split_stratify(
            test_size=args.test_ratio,
            min_reviews=3,
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
        )
        train_dataloader, val_dataloader = prepare_torch_dataloader(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        )
        model = SVDWithBias(data["num_users"], data["num_diners"], args.num_factors, mu=data["y_train"].mean())

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # train model
        best_loss = float("inf")
        for epoch in range(args.epochs):
            logger.info(f"####### Epoch {epoch} #######")

            # training
            model.train()
            tr_loss = 0.0
            for X_train, y_train in train_dataloader:
                diners, users = X_train[:, 0], X_train[:, 1]
                optimizer.zero_grad()
                y_pred = model(users, diners)
                loss = svd_loss(
                    pred=y_pred,
                    true=y_train,
                    params=[param.data for param in model.parameters()],
                    regularization=args.regularization,
                    user_idx=users,
                    diner_idx=diners,
                    num_users=data["num_users"],
                    num_diners=data["num_diners"],
                )
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X_val, y_val in val_dataloader:
                    diners, users = X_val[:, 0], X_val[:, 1]
                    y_pred = model(users, diners)
                    loss = svd_loss(
                        pred=y_pred,
                        true=y_val,
                        params=[param.data for param in model.parameters()],
                        regularization=args.regularization,
                        user_idx=users,
                        diner_idx=diners,
                        num_users=data["num_users"],
                        num_diners=data["num_diners"],
                    )

                    val_loss += loss.item()
                val_loss = round(val_loss / len(val_dataloader), 6)

            logger.info(f"Train Loss: {tr_loss}")
            logger.info(f"Validation Loss: {val_loss}")

            # todo: calculate ndcg, map at every epoch
            recommendations = model.recommend(X_train=data["X_train"], X_val=data["X_val"], filter_already_liked=True)
            for K in model.metric_at_K.keys():
                map = model.metric_at_K[K]["map"]
                ndcg = model.metric_at_K[K]["ndcg"]
                count = model.metric_at_K[K]["count"]
                logger.info(f"maP@{K}: {map} with {count} users out of all {model.num_users} users")
                logger.info(f"ndcg@{K}: {ndcg} with {count} users out of all {model.num_users} users")

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(model.state_dict(), args.model_path)
                logger.info(f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}")
            else:
                patience -= 1
                logger.info(f"Validation loss did not decrease. Patience {patience} left.")
                if patience == 0:
                    logger.info(f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss")
                    break

            # Load the best model weights
            model.load_state_dict(best_model_weights)
            logger.info("Load weight with best validation loss")

            torch.save(model.state_dict(), args.model_path)
            logger.info("Save final model")
    except:
        logger.error(traceback.format_exc())
        raise
