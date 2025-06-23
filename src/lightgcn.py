# 🔧 표준 라이브러리
import argparse
import os
import pprint
import random

# 📦 외부 라이브러리
import numpy as np
import pandas as pd
import scipy.sparse as sp

# 🔥 PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def add_arguments():
    parser_ = argparse.ArgumentParser()

    parser_.add_argument('--seed', type=int, default=0)
    parser_.add_argument('--model', type=str, default='lightgcn')

    parser_.add_argument('--data_path', type=str, default='data')
    parser_.add_argument('--edge', type=str, default='review.csv')
    parser_.add_argument('--reviewer', type=str, default='reviewer.csv')
    parser_.add_argument('--diner', type=str, default='diner.csv')

    parser_.add_argument('--batch_size_train', type=int, default=2048)
    parser_.add_argument('--batch_size_test', type=int, default=2048)
    parser_.add_argument('--topk', type=str, default='10,20,40,80')

    parser_.add_argument('--using_features', type=str, default='reviewer:@diner:',
                         help='using features')
    parser_.add_argument('--comment', type=str, default='None')

    parser_.add_argument('--embedding_size', type=int, default=64)
    parser_.add_argument('--num_layers', type=int, default=1)

    parser_.add_argument('--drop_ratio', type=float, default=0.5)

    parser_.add_argument('--lr', type=float, default=1e-04)
    parser_.add_argument('--decay', type=float, default=1e-05)

    parser_.add_argument('--alpha', type=float, default=1)
    parser_.add_argument('--beta', type=float, default=1)

    parser_.add_argument('--epochs', type=int, default=10)
    parser_.add_argument('--test_interval', type=int, default=5)

    parser_.add_argument('--gpu', type=int, default=0)
    return parser_

# Utils 관련 함수
def print_statistics(X, string):
    print('-' * 10 + string + '-' * 10)
    print(f'Avg non-zeros in row:    {X.sum(1).mean(0).item():8.4f}')
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print(f'Ratio of non-empty rows: {len(unique_nonzero_row_indice) / X.shape[0]:8.4f}')
    print(f'Ratio of non-empty cols: {len(unique_nonzero_col_indice) / X.shape[1]:8.4f}')
    print(f'Density of matrix:       {len(nonzero_row_indice) / (X.shape[0] * X.shape[1]):8.4f}')

class TrainData(Dataset):
    def __init__(self, conf, pairs, graph, num_diner, neg_sample=1):
        self.conf = conf
        self.pairs = pairs
        self.graph = graph
        self.num_diner = num_diner
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        reviewer, pos_diner, weight = self.pairs[index]
        reviewer = int(reviewer)
        pos_diner = int(pos_diner)
        all_diner = [pos_diner]

        while True:
            i = np.random.randint(self.num_diner)
            if self.graph[reviewer, i] == 0 and i not in all_diner:
                all_diner.append(i)
                if len(all_diner) == self.neg_sample + 1:
                    break

        return torch.LongTensor([reviewer]), torch.LongTensor(all_diner), torch.FloatTensor([weight])

    def __len__(self):
        return len(self.pairs)
    
class TestData(Dataset):
    def __init__(self, pairs, graph, num_reviewer, num_diner):
        self.pairs = pairs
        self.graph = graph
        self.reviewers = torch.arange(num_reviewer, dtype=torch.long).unsqueeze(dim=1)
        self.diners = torch.arange(num_diner, dtype=torch.long)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        grd = torch.from_numpy(self.graph[index].toarray()).squeeze()
        return index, grd

    def __len__(self):
        return self.graph.shape[0]

class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.edge_name = conf['edge']
        self.reviewer_name = conf['reviewer']
        self.diner_name = conf['diner']

        self.suffix = '-id'

        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        # Load datasets
        edge = pd.read_csv(os.path.join(self.path, self.edge_name))
        reviewer = pd.read_csv(os.path.join(self.path, self.reviewer_name))
        diner = pd.read_csv(os.path.join(self.path, self.diner_name))

        # Build dictionaries
        reviewers = list(reviewer['reviewer_id'].values) + list(edge['reviewer_id'].values)
        reviewers = np.unique(reviewers)
        reviewers_map = {r: i for (i, r) in enumerate(reviewers)}
        diners = list(diner['diner_idx'].values) + list(edge['diner_idx'].values)
        diners = np.unique(diners)
        diners_map = {r: i for (i, r) in enumerate(diners)}

        # Map IDs
        reviewer['reviewer_id'] = reviewer['reviewer_id'].map(reviewers_map)
        diner['diner_idx'] = diner['diner_idx'].map(diners_map)
        edge['reviewer_id'] = edge['reviewer_id'].map(reviewers_map)
        edge['diner_idx'] = edge['diner_idx'].map(diners_map)

        # Split into trn/val/test
        edge = edge.sort_values(by='reviewer_review_date')
        edge['reviewer_review_date'] = pd.to_datetime(edge['reviewer_review_date'], format='%Y-%m-%d')
        edge = edge.query('"2024-01-01"<=reviewer_review_date')
        edge_trn = edge.query('reviewer_review_date<"2024-03-01"')
        edge_val = edge.query('"2024-03-01"<=reviewer_review_date<"2024-06-01"')
        edge_test = edge.query('"2024-06-01"<=reviewer_review_date<="2024-12-31"')
        print(edge_trn['reviewer_review_date'].min(), edge_trn['reviewer_review_date'].max())
        print(edge_val['reviewer_review_date'].min(), edge_val['reviewer_review_date'].max())
        print(edge_test['reviewer_review_date'].min(), edge_test['reviewer_review_date'].max())

        # print(edge_val['type'].value_counts())
        # print(edge_test['type'].value_counts())

        self.num_reviewer, self.num_diner = len(reviewers_map), len(diners_map)
        print(f'Num_reviewer: {self.num_reviewer}, Num_diner: {self.num_diner}')

        pf_pairs_trn, pf_graph_trn = self.get_pf(edge_trn)
        pf_pairs_val, pf_graph_val = self.get_pf(edge_val)
        pf_pairs_test, pf_graph_test = self.get_pf(edge_test)
        print_statistics(pf_graph_trn, 'Trn dataset')
        print_statistics(pf_graph_val, 'Vld dataset')
        print_statistics(pf_graph_test, 'Test dataset')

        self.trn_data = TrainData(conf, pf_pairs_trn, pf_graph_trn, self.num_diner)
        self.val_data = TestData(pf_pairs_val, pf_graph_val, self.num_reviewer, self.num_diner)
        self.test_data = TestData(pf_pairs_test, pf_graph_test, self.num_reviewer, self.num_diner)

        self.trn_graph = pf_graph_trn

        self.trn_loader = DataLoader(self.trn_data, batch_size=batch_size_train, shuffle=True, num_workers=0,
                                     drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size_test, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size_test, shuffle=False, num_workers=0)

    def get_pf(self, edge):
        pf_pairs = np.array(edge[['reviewer_id', 'diner_idx', 'reviewer_review_score']])
        pf_graph = sp.coo_matrix(
            (pf_pairs[:, 2], (pf_pairs[:, 0], pf_pairs[:, 1])), shape=(self.num_reviewer, self.num_diner)).tocsr()
        return pf_pairs, pf_graph

def to_csv(out_dict, path: str):
    import csv
    is_exists = True
    header = list(out_dict.keys())
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            is_exists = False
    with open(path, 'a') as f:
        writer = csv.writer(f)

        if not is_exists:
            writer.writerow(header)

        writer.writerow(list(out_dict.values()))

# Models 관련 함수
def cal_bpr_loss(pred, weight):
    # pred: [bs, 1+neg_num]
    negs = pred[:, 1].unsqueeze(1)
    pos = pred[:, 0].unsqueeze(1)
    loss = - torch.mean(weight * torch.log(torch.sigmoid(pos - negs))) # [bs]
    return loss

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph

def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values

class LightGCN(nn.Module):
    def __init__(self, conf, trn_graph):
        super().__init__()
        self.conf = conf
        self.device = self.conf['device']
        self.embedding_size = conf['embedding_size']
        self.num_reviewer = conf['num_reviewer']
        self.num_diner = conf['num_diner']
        self.num_layers = self.conf['num_layers']
        self.trn_graph = trn_graph

        self.init_emb()
        self.get_graph_ori()
        self.get_graph()

    def init_emb(self):
        # no usage
        self.reviewer_emb = nn.Parameter(torch.FloatTensor(self.num_reviewer, self.embedding_size))
        nn.init.xavier_normal_(self.reviewer_emb)
        self.diner_emb = nn.Parameter(torch.FloatTensor(self.num_diner, self.embedding_size))
        nn.init.xavier_normal_(self.diner_emb)

    def get_graph(self):
        graph = self.trn_graph
        device = self.device
        drop_ratio = self.conf['drop_ratio']
        total_graph = sp.bmat([[sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],
                               [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))]])
        if drop_ratio != 0:
            graph = total_graph.tocoo()
            values = np_edge_dropout(graph.data, drop_ratio)
            total_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        self.graph = to_tensor(laplace_transform(total_graph)).to(device)

    def get_graph_ori(self):
        graph = self.trn_graph
        device = self.device
        total_graph = sp.bmat([[sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],
                               [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))]])
        self.graph_ori = to_tensor(laplace_transform(total_graph)).to(device)
    
    def propagate(self, test=False):
        reviewer_vec = self.reviewer_emb
        diner_vec = self.diner_emb
        return reviewer_vec, diner_vec

    def cal_loss(self, reviewer_feature, diner_feature, weight):
        pred = torch.sum(reviewer_feature * diner_feature, 2)
        bpr_loss = cal_bpr_loss(pred, weight)
        return bpr_loss

    def forward(self, batch):
        self.get_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        # weights: [bs, 1]
        users, bundles, weights = batch
        # print(users.shape)
        # print(bundles.shape)
        # print(weights.shape)
        reviewer_feature, diner_feature = self.propagate()

        reviewer_embedding = reviewer_feature[users].expand(-1, bundles.shape[1], -1)
        diner_embedding = diner_feature[bundles]
        bpr_loss = self.cal_loss(reviewer_embedding, diner_embedding, weights)

        return bpr_loss

    def evaluate(self, propagate_result, reviewers):
        reviewer_feature, diner_feature = propagate_result
        scores = torch.mm(reviewer_feature[reviewers], diner_feature.t())
        return scores

def main():
    parser = add_arguments()
    args = parser.parse_args()
    conf = args.__dict__
    print(conf)

    set_seed(conf['seed'])
    model = conf['model']
    conf['topk']: list = [int(k) for k in conf['topk'].split(',')]
    conf['using_features']: dict = \
        {k: v for k, v in zip(['reviewer', 'diner'],
                                [type_.split(':')[1].split(',') for type_ in conf['using_features'].split('@')])}
    # conf = yaml.safe_load(open('config.yaml'))

    device = torch.device(f"cuda:{conf['gpu']}" if torch.cuda.is_available() else 'cpu')
    conf['device'] = device

    dataset = Datasets(conf)
    conf['num_reviewer'] = dataset.num_reviewer
    conf['num_diner'] = dataset.num_diner
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(conf)

    model = LightGCN(conf, dataset.trn_graph).to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['decay'])
    crit = 20
    best_vld_rec, best_vld_ndcg, best_content_dict, best_content, best_loss = 0., 0., None, '', 0

    for epoch in range(1, conf['epochs'] + 1):
        model.train(True)
        pbar = tqdm(enumerate(dataset.trn_loader), total=len(dataset.trn_loader))
        cur_instance_num, bpr_loss_avg = 0., 0.

        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]

            bpr_loss = model(batch)
            bpr_loss.backward()
            optimizer.step()

            bpr_loss_scalar = bpr_loss.detach()

            bpr_loss_avg = moving_avg(bpr_loss_avg, cur_instance_num, bpr_loss_scalar, batch[0].size(0))
            cur_instance_num += batch[0].size(0)
            pbar.set_description(f'epoch: {epoch:3d} | bpr_loss: {bpr_loss_avg:8.4f}')

        if epoch % conf['test_interval'] == 0:
            metrics = {}
            metrics['val'] = test(model, dataset.val_loader, conf)
            metrics['test'] = test(model, dataset.test_loader, conf)
            content, content_dict = form_content(epoch, metrics['val'], metrics['test'], conf['topk'])
            print(content)

            if metrics['val']['recall'][crit] > best_vld_rec and metrics['val']['ndcg'][crit] > best_vld_ndcg:
                best_vld_rec = metrics['val']['recall'][crit]
                best_vld_ndcg = metrics['val']['ndcg'][crit]
                best_content = content
                best_content_dict = content_dict
                best_loss = bpr_loss_avg

    print('============================ BEST ============================')
    print(best_content)
    result_dict = {**conf, 'best_loss': best_loss.item(), **best_content_dict}
    to_csv(out_dict=result_dict, path='result/test/lightgcn/result.csv')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def moving_avg(avg, cur_num, add_value_avg, add_num):
    avg = (avg * cur_num + add_value_avg * add_num) / (cur_num + add_num)
    return avg


def form_content(epoch, val_results, test_results, ks):
    content_dict = dict(**{k: v for k, v in zip([f"val_Rec@{k}" for k in ks], [val_results['recall'][k] for k in ks])},
                        **{k: v for k, v in zip([f"val_nDCG@{k}" for k in ks], [val_results['ndcg'][k] for k in ks])},
                        **{k: v for k, v in
                           zip([f"test_Rec@{k}" for k in ks], [test_results['recall'][k] for k in ks])},
                        **{k: v for k, v in zip([f"test_nDCG@{k}" for k in ks], [test_results['ndcg'][k] for k in ks])})

    content = f'     Epoch|  Rec@{ks[0]} |  Rec@{ks[1]} |  Rec@{ks[2]} |  Rec@{ks[3]} |' \
              f' nDCG@{ks[0]} | nDCG@{ks[1]} | nDCG@{ks[2]} | nDCG@{ks[3]} |\n'
    val_content = f'{epoch:10d}|'
    val_results_recall = val_results['recall']
    for k in ks:
        val_content += f'  {val_results_recall[k]:.4f} |'
    val_results_ndcg = val_results['ndcg']
    for k in ks:
        val_content += f'  {val_results_ndcg[k]:.4f} |'
    content += val_content + '\n'
    test_content = f'{epoch:10d}|'
    test_results_recall = test_results['recall']
    for k in ks:
        test_content += f'  {test_results_recall[k]:.4f} |'
    test_results_ndcg = test_results['ndcg']
    for k in ks:
        test_content += f'  {test_results_ndcg[k]:.4f} |'
    content += test_content
    return content, content_dict


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)
    print(len(dataloader))
    for reviewers, ground_truth_u_b in dataloader:
        print(f"Batch : reviewers shape: {reviewers.shape}, ground_truth shape: {ground_truth_u_b.shape}")
        pred_b = model.evaluate(rs, reviewers.to(device))
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]
    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device,
                                                                 dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1).to('cpu'), col_indice.view(-1).to('cpu')].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]

if __name__ == '__main__':
    main()
