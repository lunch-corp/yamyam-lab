import copy
import traceback

import torch
import torch.nn as nn
from torch import optim

from loss.custom import svd_loss
from tools.parse_args import parse_args
from tools.logger import setup_logger

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


class SVDWithBias(nn.Module):

    def __init__(self, num_users, num_items, num_factors, **kwargs):
        super(SVDWithBias, self).__init__()

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


if __name__ == "__main__":
    from preprocess.preprocess import train_test_split_stratify, prepare_torch_dataloader
    args = parse_args()
    logger = setup_logger(args.log_path)

    try:
        num_factors = 32
        data = train_test_split_stratify(test_size=0.2,
                                         min_reviews=3,
                                         X_columns=["diner_idx", "reviewer_id"],
                                         y_columns=["reviewer_review_score"])
        train_dataloader, val_dataloader = prepare_torch_dataloader(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
        model = SVDWithBias(data["num_reviewers"], data["num_diners"], num_factors, mu = data["y_train"].mean())

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # train model
        best_loss = float('inf')
        for epoch in range(args.epochs):
            logger.info(f"####### Epoch {epoch} #######")

            # training
            model.train()
            tr_loss = 0.0
            for data in train_dataloader:
                X_train, y_train = data
                diners, users = X_train[:, 0], X_train[:, 1]
                optimizer.zero_grad()
                y_pred = model(users, diners)
                loss = svd_loss(pred=y_pred,
                                true=y_train,
                                params=model.parameters(),
                                regularization=args.regularization)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for data in val_dataloader:
                    X_val, y_val = data
                    diners, users = X_val[:, 0], X_val[:, 1]
                    y_pred = model(users, diners)
                    loss = svd_loss(pred=y_pred,
                                    true=y_val,
                                    params=model.parameters(),
                                    regularization=args.regularization)

                    val_loss += loss.item()
                val_loss = round(val_loss / len(val_dataloader), 6)

            logger.info(f"Train Loss: {tr_loss}")
            logger.info(f"Validation Loss: {val_loss}")

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