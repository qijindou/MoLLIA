import numpy as np
import torch
import torch.nn.functional as F 
import copy

NOISE_SCALE = 0.001
SUBSET    = 29303
ADDENDUM  = 3663

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'): 
            noise = torch.randn(m.weight.size())
            noise = noise.cuda()
            noise *= (NOISE_SCALE * m.weight.norm() / noise.norm())
            m.weight.add_(noise)

def noise_stability_sampling(conf, args, train_unlabel_ls, al_model, train_unlabel_dataloader):
    if NOISE_SCALE < 1e-8:
        uncertainty = torch.randn(len(train_unlabel_ls))
        return uncertainty
    
    uncertainty = torch.zeros(len(train_unlabel_ls)).cuda()

    diffs = torch.tensor([]).cuda()
    outputs = get_all_outputs(conf, al_model, train_unlabel_dataloader)
    original_state = {
        k: v.clone().cpu()
        for k, v in al_model.state_dict().items()
    }
    for i in range(5):
        al_model.apply(add_noise_to_weights)
        outputs_noisy = get_all_outputs(conf, al_model, train_unlabel_dataloader)

        diff_k = outputs_noisy - outputs
        for j in range(diff_k.shape[0]):
            diff_k[j,:] /= outputs[j].norm() 
        diffs = torch.cat((diffs, diff_k), dim = 1)
        al_model.load_state_dict({
            k: v.to(al_model.state_dict()[k].device)
            for k, v in original_state.items()
        })
        
    indsAll = kcenter_greedy(diffs, ADDENDUM)
    for ind in indsAll:
        uncertainty[ind] = 1

    return np.argsort(uncertainty.cpu())[-args.n_annote:]

from dppy.finite_dpps import FiniteDPP

def k_dpp(X, K):
    DPP = FiniteDPP('likelihood', **{'L_gram_factor': 1e6*X.cpu().numpy().transpose()})
    DPP.flush_samples()
    DPP.sample_mcmc_k_dpp(size=K)
    indsAll = DPP.list_of_samples[0][0]
    return indsAll

def kcenter_greedy(X, K):
    mu = torch.zeros(1, X.shape[1]).cuda()
    indsAll = []
    while len(indsAll) < K:
        if len(indsAll) == 0:
            D2 = torch.cdist(X, mu).squeeze(1)
        else:
            newD = torch.cdist(X, mu[-1:])
            newD = torch.min(newD, dim = 1)[0]
            for i in range(X.shape[0]):
                if D2[i] >  newD[i]:
                    D2[i] = newD[i]

        for i, ind in enumerate(D2.topk(1)[1]):
            D2[ind] = 0
            mu = torch.cat((mu, X[ind].unsqueeze(0)), 0)
            indsAll.append(ind)

    return indsAll

def get_all_outputs(conf, al_model, train_unlabel_dataloader):
    al_model.eval()
    outputs = torch.tensor([]).cuda()
    with torch.no_grad():
        for batch in train_unlabel_dataloader:
            input = {k: v.to(conf.al_device) for k, v in batch.items()}
            output = al_model(**input)
            out = F.softmax(output.logits, dim = 1)
            outputs = torch.cat((outputs, out), dim=0)

    return outputs
