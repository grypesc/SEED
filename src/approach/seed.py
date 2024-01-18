import copy
import random
import torch

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .gmm import GaussianMixture
from .incremental_learning import Inc_Learning_Appr

torch.backends.cuda.matmul.allow_tf32 = False


def softmax_temperature(x, dim, tau=1.0):
    return torch.softmax(x / tau, dim=dim)



class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, ftepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, ftwd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, max_experts=999, gmms=1, alpha=1.0, tau=3.0, shared=0, use_multivariate=False, use_nmc=False,
                 initialization_strategy="first", compensate_drifts=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.max_experts = max_experts
        self.model.bbs = self.model.bbs[:max_experts]
        self.gmms = gmms
        self.alpha = alpha
        self.tau = tau
        self.patience = patience
        self.use_multivariate = use_multivariate
        self.use_nmc = use_nmc
        self.ftepochs = ftepochs
        self.ftwd = ftwd
        self.compensate_drifts = compensate_drifts
        self.model.to(device)
        self.experts_distributions = []
        self.shared_layers = []
        if shared > 0:
            self.shared_layers = ["conv1_starting.weight", "bn1_starting.weight", "bn1_starting.bias", "layer1"]
            if shared > 1:
                self.shared_layers.append("layer2")
                if shared > 2:
                    self.shared_layers.append("layer3")
                    if shared > 3:
                        self.shared_layers.append("layer4")

        self.initialization_strategy = initialization_strategy

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--max-experts',
                            help='Maximum number of experts',
                            type=int,
                            default=999)
        parser.add_argument('--gmms',
                            help='Number of gaussian models in the mixture',
                            type=int,
                            default=1)
        parser.add_argument('--shared',
                            help='Number of shared blocks',
                            type=int,
                            default=0)
        parser.add_argument('--initialization-strategy',
                            help='How to initialize experts weight',
                            type=str,
                            choices=["first", "random"],
                            default="first")
        parser.add_argument('--ftepochs',
                            help='Number of epochs for finetuning an expert',
                            type=int,
                            default=100)
        parser.add_argument('--ftwd',
                            help='Weight decay for finetuning',
                            type=float,
                            default=0)
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
                            action='store_true',
                            default=True)
        parser.add_argument('--use-nmc',
                            help='Use nearest mean classifier instead of bayes',
                            action='store_true',
                            default=False)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.99)
        parser.add_argument('--tau',
                            help='softmax temperature',
                            type=float,
                            default=3.0)
        parser.add_argument('--compensate-drifts',
                            help='Drift compensation using MLP feature adaptation',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        if t < self.max_experts:
            print(f"Training backbone on task {t}:")
            self.train_backbone(t, trn_loader, val_loader)
            self.experts_distributions.append([])

        if t >= self.max_experts:
            bb_to_finetune = self._choose_backbone_to_finetune(t, trn_loader, val_loader)
            print(f"Finetuning backbone {bb_to_finetune} on task {t}:")
            self.finetune_backbone(t, bb_to_finetune, trn_loader, val_loader)

        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, trn_loader, val_loader)

    def train_backbone(self, t, trn_loader, val_loader):
        if self.initialization_strategy == "random" or t==0:
            self.model.bbs.append(self.model.bb_fun(num_classes=self.model.taskcla[t][1], num_features=self.model.num_features))
        else:
            self.model.bbs.append(copy.deepcopy(self.model.bbs[0]))
        model = self.model.bbs[t]
        model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])
        if t == 0:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                for layer_not_to_train in self.shared_layers:
                    if layer_not_to_train in name:
                        model.get_parameter(name).data = self.model.bbs[0].get_parameter(name).data
                        param.requires_grad = False

        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(t, self.wd)
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(images)
                loss = self.criterion(t, out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    out = model(images)
                    loss = self.criterion(t, out, targets)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")
        model.fc = nn.Identity()
        self.model.bbs[t] = model
        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")

    @torch.no_grad()
    def _choose_backbone_to_finetune(self, t, trn_loader, val_loader):
        self.create_distributions(t, trn_loader, val_loader)
        expert_overlap = torch.zeros(self.max_experts, device=self.device)
        for bb_num in range(self.max_experts):
            classes_in_t = self.model.taskcla[t][1]
            new_distributions = self.experts_distributions[bb_num][-classes_in_t:]
            kl_matrix = torch.zeros((len(new_distributions), len(new_distributions)), device=self.device)
            for o, old_gauss_ in enumerate(new_distributions):
                old_gauss = MultivariateNormal(old_gauss_.mu.data[0][0], covariance_matrix=old_gauss_.var.data[0][0])
                for n, new_gauss_ in enumerate(new_distributions):
                    new_gauss = MultivariateNormal(new_gauss_.mu.data[0][0], covariance_matrix=new_gauss_.var.data[0][0])
                    kl_matrix[n, o] = torch.distributions.kl_divergence(new_gauss, old_gauss)
            expert_overlap[bb_num] = torch.mean(kl_matrix)
            self.experts_distributions[bb_num] = self.experts_distributions[bb_num][:-classes_in_t]
        print(f"Expert overlap:{expert_overlap}")
        bb_to_finetune = torch.argmax(expert_overlap)
        self.model.task_offset = self.model.task_offset[:-1]
        return int(bb_to_finetune)


    def finetune_backbone(self, t, bb_to_finetune, trn_loader, val_loader):
        old_model = copy.deepcopy(self.model.bbs[bb_to_finetune])
        for name, param in old_model.named_parameters():
            param.requires_grad = False
        old_model.eval()

        model = self.model.bbs[bb_to_finetune]
        for name, param in model.named_parameters():
            param.requires_grad = True
            for layer_not_to_train in self.shared_layers:
                if layer_not_to_train in name:
                    param.requires_grad = False
        model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])
        model.to(self.device)
        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

        optimizer, lr_scheduler = self._get_optimizer(bb_to_finetune, wd=self.ftwd, milestones=[30, 60, 80])
        for epoch in range(self.ftepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    old_features = old_model(images)  # resnet with fc as identity returns features by default
                out, features = model(images, return_features=True)
                loss = self.criterion(t, out, targets, features, old_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))

            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    with torch.no_grad():
                        old_features = old_model(images)  # resnet with fc as identity returns features by default
                    out, features = model(images, return_features=True)
                    loss = self.criterion(t, out, targets, features, old_features)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        model.fc = nn.Identity()
        self.model.bbs[bb_to_finetune] = model
        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        return old_model


    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        self.model.eval()
        classes = self.model.taskcla[t][1]
        self.model.task_offset.append(self.model.task_offset[-1] + classes)
        transforms = val_loader.dataset.transform
        for bb_num in range(min(self.max_experts, t+1)):
            eps = 1e-8
            model = self.model.bbs[bb_num]
            for c in range(classes):
                c = c + self.model.task_offset[t]
                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, transforms)
                else:
                    ds = trn_loader.dataset.images[train_indices]
                    ds = ClassMemoryDataset(ds, transforms)
                loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
                from_ = 0
                class_features = torch.full((2 * len(ds), self.model.num_features), fill_value=-999999999.0, device=self.model.device)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    features = model(images)
                    class_features[from_: from_+bsz] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_+bsz: from_+2*bsz] = features
                    from_ += 2*bsz

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"
                is_ok = False
                while not is_ok:
                    try:
                        gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(self.device)
                        gmm.fit(class_features, delta=1e-3, n_iter=100)
                    except RuntimeError:
                        eps = 10 * eps
                        print(f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")
                    else:
                        is_ok = True
                        if self.use_nmc:
                            gmm.var = torch.nn.Parameter(torch.ones(self.model.num_features, device=self.device).unsqueeze(0).unsqueeze(0))

                self.experts_distributions[bb_num].append(gmm)

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        for images, targets in val_loader:
            targets = targets.to(self.device)
            # Forward current model
            features = self.model(images.to(self.device))
            hits_taw, hits_tag = self.calculate_metrics(features, targets, t)
            # Log
            total_loss = 0
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        taw_pred, tag_pred = self.predict_class_bayes(t, features)
        hits_taw = (taw_pred == targets).float()
        hits_tag = (tag_pred == targets).float()
        return hits_taw, hits_tag

    @torch.no_grad()
    def predict_class_bayes(self, t, features):
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])), fill_value=-1e8, device=features.device)
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)
        for bb_num, _ in enumerate(self.experts_distributions):
            for c, class_gmm in enumerate(self.experts_distributions[bb_num]):
                c += self.model.task_offset[bb_num]
                log_probs[:, bb_num, c] = class_gmm.score_samples(features[:, bb_num])
                mask[:, bb_num, c] = True

        from_ = self.model.task_offset[t]
        to_ = self.model.task_offset[t+1]

        # Task-Aware
        taw_log_probs = log_probs[:, :t+1, from_:to_].clone()
        taw_log_probs = softmax_temperature(taw_log_probs, dim=2, tau=self.tau)
        confidences = torch.sum(taw_log_probs, dim=1)
        taw_class_id = torch.argmax(confidences, dim=1) + self.model.task_offset[t]
        # Task-Agnostic
        log_probs = softmax_temperature(log_probs, dim=2, tau=self.tau)
        confidences = torch.sum(log_probs, dim=1) / torch.sum(mask, dim=1)
        tag_class_id = torch.argmax(confidences, dim=1)
        return taw_class_id, tag_class_id

    def criterion(self, t, outputs, targets, features=None, old_features=None):
        """Returns the loss value"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        if old_features is not None:  # Knowledge distillation loss on features
            kd_loss = nn.functional.mse_loss(features, old_features)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            return total_loss
        return ce_loss

    def _get_optimizer(self, num, wd, milestones=[60, 120, 160]):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(self.model.bbs[num].parameters(), lr=self.lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler
