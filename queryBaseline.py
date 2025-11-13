import torch
import random
import time
import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from torch.nn.functional import normalize

from noise_stability import noise_stability_sampling


def rand_select(sample_ls, train_unlabel_ls, n_pick, random_seed):
    n_pick = len(train_unlabel_ls) if n_pick > len(train_unlabel_ls) else n_pick
    if random_seed:
        random.seed(random_seed)
    else:
        random.seed(time.time())
    sample_ls += random.sample(train_unlabel_ls, n_pick)
    train_unlabel_ls = [x for x in train_unlabel_ls if x not in sample_ls]
    return sample_ls, train_unlabel_ls


def update_sample_ls(sample_ls, train_unlabel_ls, select_index):
    for idx in select_index:
        sample_ls.append(train_unlabel_ls[idx]) 
    train_unlabel_ls = [x for x in train_unlabel_ls if x not in sample_ls]
    return sample_ls, train_unlabel_ls


def get_query_index(logger, conf, args, i_al, sample_ls, train_unlabel_ls, prob_X_Y, Py_X_n=None):
    select_index = query_score(logger, conf, args, i_al, prob_X_Y, Py_X_n=None)
    return update_sample_ls(sample_ls, train_unlabel_ls, select_index) 


def query_score(logger, conf, args, i_al, prob_X_Y, Py_X_n=None):
    if args.sampling == "max_entropy":
        return max_entropy(logger, conf, args, prob_X_Y)
    elif args.sampling == "bemps":
        return bemps_mse(logger, conf, args, prob_X_Y)


def get_ns_query_index(conf, args, sample_ls, train_unlabel_ls, al_model, train_unlabel_dataloader):
    select_index = noise_stability_sampling(conf, args, train_unlabel_ls, al_model, train_unlabel_dataloader)
    return update_sample_ls(sample_ls, train_unlabel_ls, select_index) 


def get_cs_query_index(conf, args, sample_ls, train_unlabel_ls, labeled_embeddings, unlabeled_embeddings):
    select_index = []
    n_unlabeled = unlabeled_embeddings.shape[0]
    n_labeled = labeled_embeddings.shape[0]

    dist = np.full(n_unlabeled, np.inf)

    for i in range(n_labeled):
        d = np.linalg.norm(unlabeled_embeddings - labeled_embeddings[i], axis=1)
        dist = np.minimum(dist, d)

    for _ in range(args.n_annote):
        idx = np.argmax(dist)
        select_index.append(idx)

        new_center = unlabeled_embeddings[idx]
        d = np.linalg.norm(unlabeled_embeddings - new_center, axis=1)
        dist = np.minimum(dist, d)
    return update_sample_ls(sample_ls, train_unlabel_ls, select_index)


def max_entropy(logger, conf, args, prob_X_Y):
    prob_X_Y = torch.from_numpy(prob_X_Y)
    score = -torch.sum((torch.log(prob_X_Y) * prob_X_Y).double(), dim=1)
    return np.argsort(score)[-args.n_annote:]


def bemps_mse(logger, conf, args, prob_X_E_Y):
    xp_indices = random_generator_for_x_prime(prob_X_E_Y.shape[0], 0.00939)
    pr_YhThetaXp_Xp_E_Yh = prob_X_E_Y[xp_indices, :, :]

    split_prob_X_E_Y = prob_X_E_Y.split(2000, dim=0)
    rr_X_Xp = torch.cat([bemps_coremse_batch(t, pr_YhThetaXp_Xp_E_Yh) for i, t in enumerate(split_prob_X_E_Y)], dim=0)

    select_index = clustering(rr_X_Xp, 0.5, args.n_annote)
    return select_index



def kmeans(rr, k):
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(rr)
    centers = kmeans.cluster_centers_
    # find the nearest point to centers
    centroids = cdist(centers, rr).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(rr)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids


# Return the index of selected points
def clustering(rr_X_Xp, T, n_pick):
    rr_X = torch.sum(rr_X_Xp, dim=-1)
    rr_topk_X = torch.topk(rr_X, round(rr_X.shape[0] * T))
    rr_topk_X_indices = rr_topk_X.indices.cpu().detach().numpy()
    rr_X_Xp = rr_X_Xp[rr_topk_X_indices]
    rr_X_Xp = normalize(rr_X_Xp)
    rr = kmeans(rr_X_Xp, n_pick)
    rr = [rr_topk_X_indices[x] for x in rr]
    return rr


# Random generator for X prime
def random_generator_for_x_prime(x_dim, size):
    sample_indices = random.sample(range(0, x_dim), round(x_dim * size))
    return sorted(sample_indices)


def bemps_coremse_batch(prob_X_E_Y, pr_YhThetaXp_Xp_E_Yh):
    ## Pr(y|theta,x)
    pr_YThetaX_X_E_Y = prob_X_E_Y                                                                                       
    pr_ThetaL = 1 / pr_YThetaX_X_E_Y.shape[1]                                                                                                               

    ## Transpose dimension of Pr(y|theta,x), and calculate pr(theta|L,(x,y))
    pr_YThetaX_X_E_Y = pr_ThetaL * pr_YThetaX_X_E_Y                                                                                 
    pr_YThetaX_X_Y_E = torch.transpose(pr_YThetaX_X_E_Y, 1, 2)  ## transpose by dimension E and Y                           

    sum_pr_YThetaX_X_Y_1 = torch.sum(pr_YThetaX_X_Y_E, dim=-1).unsqueeze(dim=-1)                           
    pr_ThetaLXY_X_Y_E = pr_YThetaX_X_Y_E / torch.clamp(sum_pr_YThetaX_X_Y_1, min=1e-7)                 

    ## Calculate pr(y_hat)
    pr_ThetaLXY_X_1_Y_E = pr_ThetaLXY_X_Y_E.unsqueeze(dim=1)                                     
    pr_Yhat_X_Xp_Y_Yh = torch.matmul(pr_ThetaLXY_X_1_Y_E, pr_YhThetaXp_Xp_E_Yh)                                      


    ## Calculate core MSE by using unsqueeze into same dimension for pr(y_hat) and pr(y_hat|theta,x)
    pr_YhThetaXp_1_1_Xp_E_Yh = pr_YhThetaXp_Xp_E_Yh.unsqueeze(dim = 0).unsqueeze(dim = 0)
    pr_YhThetaXp_X_Y_Xp_E_Yh = pr_YhThetaXp_1_1_Xp_E_Yh.repeat(pr_Yhat_X_Xp_Y_Yh.shape[0], pr_Yhat_X_Xp_Y_Yh.shape[2], 1, 1, 1)    

    pr_Yhat_1_X_Xp_Y_Yh = pr_Yhat_X_Xp_Y_Yh.unsqueeze(dim = 0)
    pr_Yhat_E_X_Xp_Y_Yh = pr_Yhat_1_X_Xp_Y_Yh.repeat(pr_YhThetaXp_Xp_E_Yh.shape[1],1,1,1,1)
    pr_Yhat_X_Y_Xp_E_Yh = pr_Yhat_E_X_Xp_Y_Yh.transpose(0,3).transpose(0,1)                                                        

    core_mse = (pr_YhThetaXp_X_Y_Xp_E_Yh - pr_Yhat_X_Y_Xp_E_Yh).pow(2)
    core_mse_X_Y_Xp = torch.sum(core_mse.sum(dim=-1), dim=-1)
    core_mse_X_Xp_Y = torch.transpose(core_mse_X_Y_Xp, 1, 2)
    core_mse_Xp_X_Y = torch.transpose(core_mse_X_Xp_Y, 0, 1)

    ## Calculate RR
    pr_YLX_X_Y = torch.sum(pr_YThetaX_X_Y_E, dim=-1)
    rr_Xp_X_Y = pr_YLX_X_Y.unsqueeze(0) * core_mse_Xp_X_Y
    rr_Xp_X = torch.sum(rr_Xp_X_Y, dim=-1)
    rr_X_Xp = torch.transpose(rr_Xp_X, 0, 1)

    return rr_X_Xp