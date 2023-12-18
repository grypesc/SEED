import time
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from .learning_approach import Learning_Appr
import pdb
from pytorch_metric_learning import losses, miners, samplers
from torchvision.transforms import Lambda
from datasets.memory_dataset import MemoryDataset
from datasets.base_dataset import BaseDataset
from datasets.exemplars_dataset import ExemplarsDataset
from scipy.special import softmax
from scipy.stats import multivariate_normal
from datasets.exemplars_selection import override_dataset_transform


class Appr(Learning_Appr):
    """ Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
        described in https://arxiv.org/abs/1611.07725"""

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.00001, logger=None, exemplars_dataset=None, metric_lambda=0.0, margin=0.0, 
                 mining=False, icarl_lambda=1.0, sampler=None, optimizer='SGD', fix_bn=False,  metric_loss='triplet', 
                 wu_nepochs=0, wu_lr_factor=1, multi_softmax=False, no_extra_head=True, norm_features=False, cov_full=False,
                 m_per_class=8, embed_dim=64, sigma=0.01, sdc=True):

        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
        multi_softmax, wu_nepochs, wu_lr_factor, logger, exemplars_dataset)

        self.metric_lambda = metric_lambda
        self.margin = margin
        self.model_old = None
        self.mining = mining
        self.icarl_lambda = icarl_lambda
        self.sampler = sampler
        self.optimizer_method = optimizer
        self.fix_bn = fix_bn
        self.metric_loss = metric_loss
        self.no_extra_head = no_extra_head
        self.norm_features = norm_features
        self.cov_full = cov_full
        self.embed_dim = embed_dim
        self.m_per_class = m_per_class
        self.sigma = sigma
        self.sdc = sdc

        self.exemplar_means = []
        self.exemplar_covs = []

        if self.no_extra_head:
            self.model.extra_head = torch.nn.Linear(self.model.out_size, self.embed_dim).cuda()  # Extra head for metric learning

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to store up to K=2000 exemplars."
        parser.add_argument('--metric_lambda', default=1.0, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--icarl_lambda', default=0.0, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--metric_loss', default='triplet', type=str, required=False, help='(default=%(default)s)',
                            choices=['triplet', 'contrastive','multiSimilarity', 'nPairs', 'proxyNCA', 'proxyAnchor'])
        parser.add_argument('--optimizer', default='SGD', type=str, required=False, help='(default=%(default)s)')
        parser.add_argument('--margin', default=1.0, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--sigma', default=0.3, type=float, required=False, help='(default=%(default)s)')
        parser.add_argument('--m_per_class', default=8, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--embed_dim', default=64, type=int, required=False, help='(default=%(default)s)')
        parser.add_argument('--mining', action='store_true', help='(default=%(default)s)')
        parser.add_argument('--sdc', action='store_true', help='(default=%(default)s)')
        parser.add_argument('--no_extra_head', action='store_false', help='(default=%(default)s)')
        parser.add_argument('--fix_bn', action='store_true', help='(default=%(default)s)')
        parser.add_argument('--norm_features', action='store_true', help='(default=%(default)s)')
        parser.add_argument('--cov_full', action='store_true', help='(default=%(default)s)')
        parser.add_argument('--sampler', default='mPerClassSampler', help='(default=%(default)s)',
                            choices=['randomSampler', 'weightedSampler','mPerClassSampler'])
        return parser.parse_known_args(args)
    
    # Returns the optimizer
    def _get_optimizer(self):
        if self.optimizer_method == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        elif self.optimizer_method == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2

        if self.sampler == 'mPerClassSampler':
            self.MPerClassTrain(t, trn_loader, val_loader)
        else:
            self.train_main(t, trn_loader, val_loader)
        # compute mean of current tasks
        self.features_current_model = self.compute_prototypes(self.model, trn_loader, val_loader.dataset.transform)

        if t>0 and self.sdc:
            self.features_old_model = self.compute_prototypes(self.model_old, trn_loader, val_loader.dataset.transform,
                                                              output_only_features=True)
            self.semantic_drift_compesation(t)
        
        # Update model_old after computing the features on it.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()


    def compute_prototypes(self, model, trn_loader, transform, output_only_features=False):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(trn_loader.dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep the same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors, e.g. averages	
            # are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                model.eval()
                for images, targets in icarl_loader:
                    feats = model(images.to(self.device), return_features=True)[1]
                    if self.no_extra_head:
                        feats = model.extra_head(feats)
                    # normalize
                    if self.norm_features:
                        extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    else:
                        extracted_features.append(feats)
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            if output_only_features:
                 return extracted_features.cpu()
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                #if self.norm_features:
                #    cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm() # TODO: recheck normalization
                #else:
                cls_feats_mean = cls_feats.mean(0)

                self.exemplar_means.append(cls_feats_mean)
                # self.exemplar_covs.append(np.cov(cls_feats.T.cpu().numpy())) # TODO: 
        return extracted_features.cpu()

    def MPerClassTrain(self, t, trn_loader, val_loader, balanced=False):    
        
        ds = trn_loader.dataset
        targets = ds.labels
        sampler = samplers.MPerClassSampler(targets, m=self.m_per_class)
        loader = DataLoader(ds,
                        batch_size=trn_loader.batch_size,
                        sampler=sampler,
                        drop_last=True,
                        num_workers=trn_loader.num_workers,
                        pin_memory=trn_loader.pin_memory)
        self.train_main(t, loader, val_loader)

    # from LwF: Runs a single epoch
    def train_epoch(self, t, trn_loader):
        if t>0 and self.fix_bn:
            self.model.eval()
        else:
            self.model.train()

        for images, targets in trn_loader:
            # Forward old model
            if t > 0 and self.icarl_lambda>0:
                _, embeddings_old = self.model_old(images.cuda(), return_features=True)
                if self.no_extra_head:
                    embeddings_old = self.model_old.extra_head(embeddings_old)
            else:
                embeddings_old = None
            # Forward current model
            _, embeddings = self.model(images.cuda(), return_features=True)
            if self.no_extra_head:
                embeddings = self.model.extra_head(embeddings)
            loss = self.criterion(t, embeddings, targets.cuda(), embeddings_old)
            if self.metric_lambda>0:
                loss += self.metric_lambda * self.criterion_metric(t, embeddings, targets.cuda())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
    
    # Contains the epochs loop
    def train_main(self, t, trn_loader, val_loader):
        best_loss = np.inf
        best_model = self.model.get_copy()
        lr = self.lr
        patience = self.lr_patience

        self.pre_train_process(t, trn_loader)

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            train_loss, train_acc, _ = self.eval_batch(t, trn_loader)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")

            # Valid
            valid_loss, valid_acc, _ = self.eval_batch(t, val_loader)
            print(' Valid: loss={:.3f}, TAw acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

        self.post_train_process(t, trn_loader)

        return
    
    # Evaluate training and valid losses
    def eval_batch(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.cuda()
                # Forward old model
                if t > 0 and self.icarl_lambda > 0:
                    _, feats_old = self.model_old(images, return_features=True)
                    if self.no_extra_head:
                        feats_old = self.model.extra_head(feats_old)
                else:
                    feats_old = None
                # Forward current model
                _, feats = self.model(images, return_features=True)
                if self.no_extra_head:
                    feats = self.model.extra_head(feats)
                loss = self.criterion(t, feats, targets.cuda(), feats_old)
                # during training, the usual accuracy is computed on the outputs
                # Log
                total_loss += loss.item() * len(targets)
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # NMC classifier evaluation
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.cuda()
                # Forward old model
                if t > 0 and self.icarl_lambda > 0:
                    _, feats_old = self.model_old(images, return_features=True)
                    if self.no_extra_head:
                        feats_old = self.model.extra_head(feats_old)
                else:
                    feats_old = None
                # Forward current model
                _, feats = self.model(images, return_features=True)
                if self.no_extra_head:
                    feats = self.model.extra_head(feats)
                loss = self.criterion(t, feats, targets.cuda(), feats_old)
                # during training, the usual accuracy is computed on the outputs
                hits_taw, hits_tag = self.classify(t, feats, targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()

                # Log
                total_loss += loss.item() * len(targets)
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # classification and distillation terms from Alg. 3. -- original formulation has no trade-off parameter
    def criterion(self, t, outputs, targets, outputs_old):
        # Classification loss for new classes
        loss = torch.zeros(1).cuda()
        # Distilation loss for old classes
        if t > 0 and self.icarl_lambda>0:
            # The E-LwF loss here
            loss += self.icarl_lambda * torch.dist(outputs, outputs_old)
        return loss

    def criterion_metric(self, t, outputs, targets):
        if self.mining:
            # Set the mining function
            miner = miners.MultiSimilarityMiner(epsilon=0.1)
            hard_pairs = miner(outputs, targets)
        else:
            hard_pairs = None
        
        if self.metric_loss == 'triplet':
            loss_func = losses.TripletMarginLoss(margin=self.margin)
        elif self.metric_loss == 'contrastive':
            loss_func = losses.ContrastiveLoss(pos_margin=self.margin)
        elif self.metric_loss == 'multiSimilarity':
            loss_func = losses.MultiSimilarityLoss(alpha = 2.0, beta = 40.0)
        elif self.metric_loss == 'nPairs':
            loss_func = losses.NPairsLoss()
        # elif self.metric_loss == 'proxyNCA':
        #     loss_func = losses.ProxyNCALoss(num_classes = 100, embedding_size = self.model.out_size)
        # elif self.metric_loss == 'proxyAnchor':
        #     loss_func = losses.ProxyAnchorLoss(num_classes = 100, embedding_size = self.model.out_size, margin = 0.1, alpha = 32)
        loss = loss_func(outputs, targets, hard_pairs)

        return loss
    
    # Algorithm 1: iCaRL Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        if self.norm_features:
            features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        if self.cov_full:
            W = []
            for i in range(means.shape[2]):
                mean = means[0,:,i].cpu().numpy()
                cov = self.exemplar_covs[i]
                W_tmp = multivariate_normal.pdf(features[:,:,0].cpu(), mean=mean, cov=cov)
                W.append(W_tmp)
            dists = -np.asarray(W).T
        else:
            dists = (features - means).pow(2).sum(1).squeeze()
            dists = dists.cpu().numpy()

        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task].numpy()
        offset = self.model.task_offset[task].numpy()
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.numpy()).sum()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.numpy()).sum()
        return hits_taw, hits_tag
    
    # Compute semantic drift compesation
    def semantic_drift_compesation(self, t):    
        DY = self.features_current_model-self.features_old_model
        means = torch.stack(self.exemplar_means).cpu()
        num_old_cls = sum(self.model.task_cls[:t])
        means_old = means[:num_old_cls]
        distance = np.sum((np.tile(self.features_old_model[None,:,:],[means_old.shape[0],1,1])-np.tile(means_old[:,None,:],[1,self.features_old_model.shape[0],1]))**2,axis=2)
        W = np.exp(-distance/(2*self.sigma **2)) + 1e-5
        W_norm = W/np.tile(np.sum(W,axis=1)[:,None],[1,W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:,:,None],[1,1,DY.shape[1]])*np.tile(DY[None,:,:],[W.shape[0],1,1]),axis=1)
        means_update = means_old + displacement
        for i in range(num_old_cls):
            self.exemplar_means[i] = means_update[i].cuda()
        