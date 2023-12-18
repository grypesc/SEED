import copy
import random
import torch
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet32 import resnet32
from .incremental_learning import Inc_Learning_Appr

torch.backends.cuda.matmul.allow_tf32 = False

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10, K=3, S=64, alpha=1.0, adapt=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.K = K
        self.S = S
        self.adapt = adapt
        self.alpha = alpha
        self.patience = patience
        self.old_model = None
        self.model = resnet32(num_features=S)
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.prototypes = {}
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.distance_metric = "L2"


    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N',
                            help='Number of learners',
                            type=int,
                            default=10)
        parser.add_argument('--K',
                            help='number of learners sampled for task',
                            type=int,
                            default=3)
        parser.add_argument('--S',
                            help='leatent space size',
                            type=int,
                            default=64)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=1.0)
        parser.add_argument('--adapt',
                            help='Adapt prototypes',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print("### Training backbone ###")
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        if t > 0 and self.adapt:
            print("### Adapting prototypes ###")
            self.adapt_prototypes(t, trn_loader, val_loader)
        print("### Creating new prototypes ###")
        self.create_prototypes(t, trn_loader, val_loader, num_classes_in_t)


    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')
        head = nn.Linear(self.S, num_classes_in_t)
        head.to(self.device)
        distiller = nn.Linear(self.S, self.S)
        distiller.to(self.device)
        parameters = list(self.model.parameters()) + list(head.parameters()) + list(distiller.parameters())
        optimizer, lr_scheduler = self.get_optimizer(parameters, self.wd)
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            self.model.train()
            head.train()
            distiller.train()
            for images, targets in trn_loader:
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                features = self.model(images)
                old_features = None
                if t > 0:
                    with torch.no_grad():
                        old_features = self.old_model(images)
                out = head(features)
                loss = self.criterion(t, out, targets, distiller(features), old_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            self.model.eval()
            head.eval()
            distiller.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    features = self.model(images)
                    old_features = None
                    if t > 0:
                        old_features = self.old_model(images)
                    out = head(features)
                    loss = self.criterion(t, out, targets, features, old_features)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")
        self.model.fc = nn.Identity()


    @torch.no_grad()
    def create_prototypes(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Create distributions for task t"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        model = self.model
        for c in range(num_classes_in_t):
            c = c + self.task_offset[t]
            train_indices = torch.tensor(trn_loader.dataset.labels) == c
            # Uncomment to add valid set to distributions
            # val_indices = torch.tensor(val_loader.dataset.labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
                # val_images = list(compress(val_loader.dataset.images, val_indices))
                # ds = ClassDirectoryDataset(train_images + val_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                # ds = np.concatenate((trn_loader.dataset.images[train_indices], val_loader.dataset.images[val_indices]), axis=0)
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                features = model(images)
                class_features[from_: from_+bsz] = features
                features = model(torch.flip(images, dims=(3,)))
                class_features[from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # Calculate prototype
            centroid = class_features.mean(dim=0)
            self.prototypes[c] = centroid


    def adapt_prototypes(self, t, trn_loader, val_loader):
        self.model.eval()
        self.old_model.eval()
        # adapter = nn.Sequential(nn.Linear(self.S, 256), nn.ReLU(), nn.Linear(256, self.S))
        adapter = nn.Sequential(nn.Linear(self.S, self.S))
        adapter.to(self.device)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_prototypes = copy.deepcopy(self.prototypes)
        for epoch in range(self.nepochs):
            adapter.train()
            train_loss, valid_loss = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    target = self.model(images)
                    old_features = self.old_model(images)
                adapted_features = adapter(old_features)
                loss = torch.nn.functional.mse_loss(adapted_features, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            adapter.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    target = self.model(images)
                    old_features = self.old_model(images)
                    adapted_features = adapter(old_features)
                    loss = torch.nn.functional.mse_loss(adapted_features, target)
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} ")

        with torch.no_grad():
            adapter.eval()
            for c, prototype in self.prototypes.items():
                self.prototypes[c] = adapter(prototype)

            # Evaluation
            for (subset, loaders) in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
                old_dist, new_dist = [], []
                class_images = np.concatenate([dl.dataset.images for dl in loaders])
                labels = np.concatenate([dl.dataset.labels for dl in loaders])

                for c in list(old_prototypes.keys()):
                    train_indices = torch.tensor(labels) == c
                    ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
                    from_ = 0
                    class_features = torch.full((2 * len(ds), self.S), fill_value=-999999999.0, device=self.device)
                    for images in loader:
                        bsz = images.shape[0]
                        images = images.to(self.device)
                        features = self.model(images)
                        class_features[from_: from_+bsz] = features
                        features = self.model(torch.flip(images, dims=(3,)))
                        class_features[from_+bsz: from_+2*bsz] = features
                        from_ += 2*bsz

                    # Calculate distance to old prototype
                    old_dist.append(torch.cdist(class_features, old_prototypes[c].unsqueeze(0)).mean())
                    new_dist.append(torch.cdist(class_features, self.prototypes[c].unsqueeze(0)).mean())

                old_dist = torch.stack(old_dist)
                new_dist = torch.stack(new_dist)
                print(f"Old {subset} distance: {old_dist.mean():.2f} ± {old_dist.std():.2f}")
                print(f"New {subset} distance: {new_dist.mean():.2f} ± {new_dist.std():.2f}")


    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform nearest mean classification based on distance to prototypes """
        self.model.eval()
        prototypes = torch.stack(list(self.prototypes.values()))
        tag_acc = Accuracy("multiclass", num_classes=prototypes.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device)
            features = self.model(images)
            if self.distance_metric == "L2":
                dist = torch.cdist(features, prototypes)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            else: # cosine
                pass
                # cos_sim = F.normalize(features) @ F.normalize(prototypes).T
                # tag_preds = torch.argmax(cos_sim, dim=1)
                # taw_preds = torch.argmax(cos_sim[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def criterion(self, t, outputs, targets, features, old_features=None):
        """Returns the loss value"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        if old_features is None:
            return ce_loss
        kd_loss = nn.functional.mse_loss(features, old_features)
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        return total_loss


    def get_optimizer(self, parameters, wd, milestones=[30, 60, 90]):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=[30, 60, 90]):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=0.001, weight_decay=0, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler