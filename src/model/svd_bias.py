import copy
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from candidate.near import NearCandidateGenerator
from evaluation.metric import ranked_precision, ranking_metrics_at_k
from loss.custom import svd_loss
from tools.logger import setup_logger
from tools.parse_args import parse_args
from tools.utils import convert_tensor, get_user_locations

# from tools.candidate import get_diner_nearby_candidates

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
        embed_user = self.embed_user(user_idx) # batch_size * num_factors
        embed_item = self.embed_item(item_idx) # batch_size * num_factors
        user_bias = self.user_bias(user_idx) # batch_size * 1
        item_bias = self.item_bias(item_idx) # batch_size * 1
        output = (embed_user * embed_item).sum(axis=1) + user_bias.squeeze() + item_bias.squeeze() + self.mu # batch_size * 1
        return output

    def recommend(self, X_train, X_val, nearby_candidates, top_K = [3, 5, 7, 10, 20], filter_already_liked=True):

        self.map = 0.
        self.ndcg = 0.

        train_liked = convert_tensor(X_train, dict)
        val_liked = convert_tensor(X_val, list)
        user_locations = get_user_locations(X_val)
        res = {}
        metric_at_K = {k:{"map":0, "ndcg":0, "count":0, "ranked_prec":0} for k in top_K}
        for user in range(self.num_users):
            item_idx = torch.arange(self.num_items)
            user_idx = torch.tensor([user]).repeat(self.num_items)

            # diner_ids visited by user in validation dataset
            locations = user_locations[user]

            # calculate one user's predicted scores for all item_ids
            with torch.no_grad():
                scores = self.forward(user_idx, item_idx)

            # filter item_id in train dataset
            if filter_already_liked:
                user_liked_items = train_liked[user]
                for already_liked_item_id in user_liked_items.keys():
                    scores[already_liked_item_id] = -float('inf') # not recommend already chosen item_id

            # calculate metric
            val_liked_item_id = np.array(val_liked[user])
            for K in top_K:
                if len(val_liked_item_id) < K:
                    continue

                # recommendations for all item pools
                pred_liked_item_id = torch.topk(scores, k=K).indices.detach().cpu().numpy()
                metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                metric_at_K[K]["map"] += metric["ap"]
                metric_at_K[K]["ndcg"] += metric["ndcg"]
                metric_at_K[K]["count"] += 1

                for location in locations:
                    # filter only near diner
                    near_diner = np.array(nearby_candidates[location])
                    near_diner_score = np.array([scores[i].item() for i in near_diner])

                    # sort indices using predicted score
                    indices = np.argsort(near_diner_score)[::-1]
                    pred_near_liked_item_id = near_diner[indices][:K]
                    metric_at_K[K]["ranked_prec"] += ranked_precision(location, pred_near_liked_item_id)



                # store recommendation result when K=20
                if K == 20:
                    res[user] = pred_liked_item_id
        for K in top_K:
            metric_at_K[K]["map"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ndcg"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ranked_prec"] /= X_val.shape[0]
        self.metric_at_K = metric_at_K
        return res


if __name__ == "__main__":
    from preprocess.preprocess import (
        prepare_torch_dataloader,
        train_test_split_stratify,
    )
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
        data = train_test_split_stratify(test_size=args.test_ratio,
                                         min_reviews=3,
                                         X_columns=["diner_idx", "reviewer_id"],
                                         y_columns=["reviewer_review_score"])
        train_dataloader, val_dataloader = prepare_torch_dataloader(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        model = SVDWithBias(data["num_users"], data["num_diners"], args.num_factors, mu = data["y_train"].mean())

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # get near 1km diner_ids
        candidate_generator = NearCandidateGenerator()
        near_diners = candidate_generator.get_near_candidates_for_all_diners(max_distance_km=1)
        # convert diner_ids
        diner_mapping = data["diner_mapping"]
        nearby_candidates_mapping = {}
        for ref_id, nearby_id in near_diners.items():
            nearby_id_mapping = [diner_mapping[diner_id] for diner_id in nearby_id]
            nearby_candidates_mapping[diner_mapping[ref_id]] = nearby_id_mapping

        # train model
        best_loss = float('inf')
        for epoch in range(args.epochs):
            logger.info(f"####### Epoch {epoch} #######")

            # training
            model.train()
            tr_loss = 0.0
            for (X_train, y_train) in train_dataloader:
                diners, users = X_train[:, 0], X_train[:, 1]
                optimizer.zero_grad()
                y_pred = model(users, diners)
                loss = svd_loss(pred=y_pred,
                                true=y_train,
                                params=[param.data for param in model.parameters()],
                                regularization=args.regularization,
                                user_idx=users,
                                diner_idx=diners,
                                num_users=data["num_users"],
                                num_diners=data["num_diners"])
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for (X_val, y_val) in val_dataloader:
                    diners, users = X_val[:, 0], X_val[:, 1]
                    y_pred = model(users, diners)
                    loss = svd_loss(pred=y_pred,
                                    true=y_val,
                                    params=[param.data for param in model.parameters()],
                                    regularization=args.regularization,
                                    user_idx=users,
                                    diner_idx=diners,
                                    num_users=data["num_users"],
                                    num_diners=data["num_diners"])

                    val_loss += loss.item()
                val_loss = round(val_loss / len(val_dataloader), 6)

            logger.info(f"Train Loss: {tr_loss}")
            logger.info(f"Validation Loss: {val_loss}")

            recommendations = model.recommend(
                X_train=data["X_train"],
                X_val=data["X_val"],
                nearby_candidates=nearby_candidates_mapping,
                filter_already_liked=True
            )
            maps = []
            ndcgs = []
            ranked_precs = []
            for K in model.metric_at_K.keys():
                map = round(model.metric_at_K[K]["map"], 5)
                ndcg = round(model.metric_at_K[K]["ndcg"], 5)
                ranked_prec = round(model.metric_at_K[K]["ranked_prec"], 5)
                count = model.metric_at_K[K]["count"]
                logger.info(f"maP@{K}: {map} with {count} users out of all {model.num_users} users")
                logger.info(f"ndcg@{K}: {ndcg} with {count} users out of all {model.num_users} users")
                logger.info(f"ranked precision@{K}: {ranked_prec}")

                maps.append(str(map))
                ndcgs.append(str(ndcg))
                ranked_precs.append(str(ranked_prec))

            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")
            logger.info(f"ranked_prec result: {'|'.join(ranked_precs)}")

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