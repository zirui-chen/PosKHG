from load_data import Data
from models import *
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch
import argparse
device = torch.device('cuda:0')

class Experiment:
    def __init__(self, num_iterations, batch_size, lr, dr, rdim, edim, m, K, max_ary):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.lr, self.dr = lr, dr
        self.rdim, self.edim = rdim, edim
        self.max_ary = max_ary
        self.m = m
        self.K = K
        self.kwargs = {'drop_role':args.drop_role, 'drop_ent':args.drop_ent}

    def get_batch(self, er_vocab, er_vocab_pairs, idx, miss_ent_domain):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = torch.zeros((len(batch), len(d.ent2id)), device=device)
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        batch = torch.tensor(batch, dtype=torch.long).to(device)
        r_idx = batch[:, 0]
        e_idx = batch[:, [i for i in range(1, batch.shape[1]) if i != miss_ent_domain]]
        return batch, targets, r_idx, e_idx

    def get_test_batch(self, test_data_idxs, idx, miss_ent_domain):
        batch = torch.tensor(test_data_idxs[idx:idx+self.batch_size], dtype=torch.long).to(device)
        r_idx = batch[:, 0]
        e_idx = batch[:, [i for i in range(1, batch.shape[1]) if i != miss_ent_domain]]
        return batch, r_idx, e_idx

    def evaluate(self, model, test_data_idxs, ary_test):
        hits, ranks = [], []
        group_hits, group_ranks = [[] for _ in range(2, self.max_ary + 1)], [[] for _ in range(2, self.max_ary + 1)]
        for _ in [1, 3, 10]:
            hits.append([])
            for h in group_hits:
                h.append([])

        for ary in ary_test:
            if len(test_data_idxs[ary-2]) > 0:
                for miss_ent_domain in range(1, ary+1):
                    er_vocab = d.all_er_vocab_list[ary-2][miss_ent_domain-1]
                    for i in range(0, len(test_data_idxs[ary-2]), self.batch_size):
                        data_batch, r_idx, e_idx = self.get_test_batch(test_data_idxs[ary-2], i, miss_ent_domain)
                        ents_idx = data_batch[:, 1:]
                        pred = model.forward(r_idx, ents_idx, miss_ent_domain)

                        for j in range(data_batch.shape[0]):
                            er_vocab_key = []
                            for k0 in range(data_batch.shape[1]):
                                er_vocab_key.append(data_batch[j][k0].item())
                            er_vocab_key[miss_ent_domain] = miss_ent_domain * 111111

                            filt = er_vocab[tuple(er_vocab_key)]
                            target_value = pred[j, data_batch[j][miss_ent_domain]].item()
                            pred[j, filt] = -1e10
                            pred[j, data_batch[j][miss_ent_domain]] = target_value

                        sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                        sort_idxs = sort_idxs.cpu().numpy()

                        for j in range(pred.shape[0]):
                            rank = np.where(sort_idxs[j] == data_batch[j][miss_ent_domain].item())[0][0]
                            ranks.append(rank + 1)
                            group_ranks[ary - 2].append(rank + 1)
                            for id, hits_level in enumerate([1, 3, 10]):
                                if rank + 1 <= hits_level:
                                    hits[id].append(1.0)
                                    group_hits[ary - 2][id].append(1.0)
                                else:
                                    hits[id].append(0.0)
                                    group_hits[ary - 2][id].append(0.0)

        t_MRR = np.mean(1. / np.array(ranks))
        t_hit10, t_hit3, t_hit1 = np.mean(hits[2]), np.mean(hits[1]), np.mean(hits[0])
        group_MRR = [np.mean(1. / np.array(x)) for x in group_ranks]
        group_HitsRatio = [[] for _ in range(2, self.max_ary + 1)]
        for i in range(0, len(group_HitsRatio)):
            for id in range(0, len([1, 3, 10])):
                group_HitsRatio[i].append(np.mean(group_hits[i][id]))
        return t_MRR, t_hit10, t_hit3, t_hit1, group_MRR, group_HitsRatio

    def train_and_eval(self):

        model = RAM(self.K, len(d.rel2id), len(d.ent2id), self.rdim, self.edim, self.m, self.max_ary, device, **self.kwargs)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.dr:
            scheduler = ExponentialLR(opt, self.dr)

        print('Training Starts...')
        test_mrr, test_hits = [], []
        best_valid_iter = 0
        best_valid_metric = {'valid_mrr': -1, 'test_mrr': -1, 'test_hit1': -1, 'test_hit3': -1, 'test_hit10': -1, 'group_test_mrr':[], 'group_test_hits':[]}

        ary_er_vocab_list = []
        ary_er_vocab_pair_list = [[] for _ in range(2, self.max_ary+1)]
        for ary in range(2, self.max_ary+1):
            ary_er_vocab_list.append(d.train_er_vocab_list[ary-2])
            for miss_ent_domain in range(1, ary+1):
                ary_er_vocab_pair_list[ary-2].append(list(d.train_er_vocab_list[ary-2][miss_ent_domain-1].keys()))

        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            print('\nEpoch %d starts training...' % it)
            for ary in args.ary_list:
                for er_vocab_pairs in ary_er_vocab_pair_list[ary-2]:
                    np.random.shuffle(er_vocab_pairs)
                for miss_ent_domain in range(1, ary+1):
                    er_vocab = ary_er_vocab_list[ary-2][miss_ent_domain-1]
                    er_vocab_pairs = ary_er_vocab_pair_list[ary-2][miss_ent_domain-1]
                    for j in range(0, len(er_vocab_pairs), self.batch_size):
                        data_batch, label, rel_idx, ent_idx = self.get_batch(er_vocab, er_vocab_pairs, j, miss_ent_domain)
                        ents_idx = data_batch[:, 1:]
                        pred = model.forward(rel_idx, ents_idx, miss_ent_domain)
                        pred = pred.to(device)
                        loss = model.loss(pred, label)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        losses.append(loss.item())

            if self.dr:
                scheduler.step()
            print('Epoch %d train, Loss=%f' % (it, np.mean(losses)))

            if it % param['eval_step'] == 0:
                model.eval()
                with torch.no_grad():
                    print('\n ~~~~~~~~~~~~~ Valid ~~~~~~~~~~~~~~~~')
                    v_mrr, _, _, _, _, _ = self.evaluate(model, d.valid_facts, args.ary_list)

                    print('~~~~~~~~~~~~~ Test ~~~~~~~~~~~~~~~~')
                    t_mrr, t_hit10, t_hit3, t_hit1, group_mrr, group_hits = self.evaluate(model, d.test_facts, args.ary_list)

                    if v_mrr >= best_valid_metric['valid_mrr']:
                        best_valid_iter = it
                        best_valid_metric['valid_mrr'] = v_mrr
                        best_valid_metric['test_mrr'] = t_mrr
                        best_valid_metric['test_hit10'], best_valid_metric['test_hit3'], best_valid_metric['test_hit1'] = t_hit10, t_hit3, t_hit1
                        best_valid_metric['group_test_hits'] = group_hits
                        best_valid_metric['group_test_mrr'] = group_mrr
                        print('Epoch=%d, Valid MRR increases.' % it)
                    else:
                        print('Valid MRR didnt increase for %d epochs, Best_MRR=%f' % (it-best_valid_iter, best_valid_metric['test_mrr']))

                    if it - best_valid_iter >= param['valid_patience'] or it == self.num_iterations:
                        print('++++++++++++ Early Stopping +++++++++++++')
                        for i, ary in enumerate(args.ary_list):
                            print('Testing Arity:%d, MRR=%f, Hits@10=%f, Hits@3=%f, Hits@1=%f' % (
                            ary, best_valid_metric['group_test_mrr'][i], best_valid_metric['group_test_hits'][i][2],
                            best_valid_metric['group_test_hits'][i][1], best_valid_metric['group_test_hits'][i][0]))

                        print('Best epoch %d' % best_valid_iter)
                        print('Hits @10: {0}'.format(best_valid_metric['test_hit10']))
                        print('Hits @3: {0}'.format(best_valid_metric['test_hit3']))
                        print('Hits @1: {0}'.format(best_valid_metric['test_hit1']))
                        print('Mean reciprocal rank: {0}'.format(best_valid_metric['test_mrr']))

                        return best_valid_metric['test_mrr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB-AUTO", nargs="?", help="FB-AUTO/JF17K/WikiPeople/WN18/FB15k.")
    parser.add_argument("--num_iterations", type=int, default=20000, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=64, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="Decay rate.")
    parser.add_argument("--K", type=int, default=10, nargs="?", help="Latent role space size.")
    parser.add_argument("--rdim", type=int, default=50, nargs="?", help="Embedding dimensionality.")
    parser.add_argument("--m", type=int, default=2, nargs="?", help="Multiplicity of entity embedding.")
    parser.add_argument("--drop_role", type=float, default=0.2, nargs="?", help="Dropout.")
    parser.add_argument("--drop_ent", type=float, default=0.4, nargs="?", help="Dropout.")
    parser.add_argument("--eval_step", type=int, default=1, nargs="?", help="Evaluation step.")
    parser.add_argument("--valid_patience", type=int, default=10, nargs="?", help="Valid patience.")
    parser.add_argument("-ary", "--ary_list", type=int, action='append', help="List of arity for train and test")
    args = parser.parse_args()
    args.ary_list = [2, 4, 5]

    param = {}
    param['dataset'] = args.dataset
    param['num_iterations'], param['eval_step'], param['valid_patience'] = args.num_iterations, args.eval_step, args.valid_patience
    param['batch_size'] = args.batch_size
    param['K'] = args.K
    param['lr'], param['dr'] = args.lr, args.dr
    param['rdim'], param['edim'] = args.rdim, args.rdim*args.m
    param['m'] = args.m
    param['drop_role'], param['drop_ent'] = args.drop_role, args.drop_ent

    # Reproduce Results
    torch.backends.cudnn.deterministic = True
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    data_dir = "./data/%s/" % param['dataset']
    print('\nLoading data...')
    d = Data(data_dir=data_dir)

    Exp = Experiment(num_iterations=param['num_iterations'], batch_size=param['batch_size'], lr=param['lr'], dr=param['dr'],
                     rdim=param['rdim'], edim=param['edim'], m=param['m'], K=param['K'], max_ary=d.max_ary)
    Exp.train_and_eval()