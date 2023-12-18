import copy
import pickle
import random
import numpy as np
import torch

from argparse import ArgumentParser
from itertools import compress
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from .gmm import GaussianMixture
from .incremental_learning import Inc_Learning_Appr


class DistributionsAnalyzer:
    def __init__(self):
        self.total_samples = 0
        self.class_dict = {}
        self.class_size = {}

    def add(self, c, features):
        self.class_dict[c] = features
        self.class_size[c] = len(features)

    def plot(self):
        data = [f for f in self.class_dict.values()]
        data = torch.cat(data, dim=0)
        data = np.array(data)
        model = PCA(n_components=3)
        data = model.fit_transform(data)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        from_ = 0
        class_to_visualize = 51
        for i, c in enumerate(self.class_dict.keys()):
            lol = data[from_:from_ + self.class_size[c]][:30]
            ax.scatter(lol[:, 0], lol[:, 1], lol[:, 2], c="b" if c != class_to_visualize else "r")
            from_ += self.class_size[c]
        plt.show()
        print("lol")


class ClassMemoryDataset(torch.utils.data.Dataset):
    """ Dataset consisting of samples of only one class """
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        image = self.transforms(image)
        return image


class ClassDirectoryDataset(torch.utils.data.Dataset):
    """ Dataset consisting of samples of only one class loaded from disc """
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transforms(image)
        return image


class DistributionDataset(torch.utils.data.Dataset):
    """ Dataset that samples from learned distributions to train head """
    def __init__(self, distributions, samples, task_cla, tasks_known):
        self.distributions = distributions
        self.samples = samples
        self.task_cla = task_cla
        task_offset = [c[1] for c in task_cla]
        for i, c in enumerate(task_offset[1:]):
            task_offset[i+1] += task_offset[i]
        self.task_offset = [0] + task_offset
        self.tasks_known = tasks_known

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        t = random.randint(0, self.tasks_known)
        target = random.randint(self.task_offset[t], self.task_offset[t+1]-1)
        val = self.distributions[target].sample(1)[0].squeeze(0)
        return val, target


class WarmUpScheduler(nn.Module):
    """Warm-up and exponential decay chain scheduler. If warm_up_iters > 0 than warm-ups linearly for warm_up_iters iterations.
    Then it decays the learning rate every epoch. It is a good idea to set warm_up_iters as total number of samples in epoch / batch size"""

    def __init__(self, optimizer, warm_up_iters=0, lr_decay=0.97):
        super().__init__()
        self.total_steps, self.warm_up_iters = 0, warm_up_iters
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-6, total_iters=warm_up_iters) if warm_up_iters else None
        self.decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay, last_epoch=-1)

    def step_iter(self):
        self.total_steps += 1
        if self.warmup_scheduler:
            self.warmup_scheduler.step()

    def step_epoch(self):
        if self.total_steps > self.warm_up_iters:
            self.decay_scheduler.step()


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, gmms=1, use_multivariate=True, use_head=False, remove_outliers=False, load_distributions=False, save_distributions=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.gmms = gmms
        self.patience = patience
        self.use_multivariate = use_multivariate
        self.use_head = use_head
        self.remove_outliers = remove_outliers
        self.load_distributions = load_distributions
        self.save_distributions = save_distributions
        self.model.to(device)
        self.task_distributions = []
        self.analyzer = DistributionsAnalyzer()

        if load_distributions:
            with open(f"distributions.pickle", 'rb') as f:
                data_file = pickle.load(f)
                self.task_distributions = data_file["distributions"]

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--gmms',
                            help='Number of gaussian models in the mixture',
                            type=int,
                            default=1)
        parser.add_argument('--patience',
                            help='Early stopping',
                            type=int,
                            default=5)
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
                            action='store_true',
                            default=False)
        parser.add_argument('--use-head',
                            help='Use trainable head instead of Bayesian inference',
                            action='store_true',
                            default=False)
        parser.add_argument('--remove-outliers',
                            help='Remove class outliers before creating distribution',
                            action='store_true',
                            default=False)
        parser.add_argument('--load-distributions',
                            help='Load distributions from a pickle file',
                            action='store_true',
                            default=False)
        parser.add_argument('--save-distributions',
                            help='Save distributions to a pickle file',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        # Train backbone
        if t == 0:
            print(f"Training backbone on task {t}:")
            self.train_backbone(t, trn_loader, val_loader)

        # Create distributions
        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, trn_loader, val_loader)

        # Train head
        if self.use_head:
            print(f"Training head for task {t}:")
            self.train_head(t)

        # Dump distributions
        if self.save_distributions:
            with open(f"distributions.pickle", 'wb') as f:
                pickle.dump({"distributions": self.task_distributions}, f)

    def train_head(self, t):
        self.model.bb.eval()
        self.model.freeze_backbone()
        self.model.replace_head(len(self.task_distributions))
        self.model.head.train()
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=self.lr, weight_decay=0)
        scheduler = WarmUpScheduler(optimizer, 100, 0.85)
        ds = DistributionDataset(self.task_distributions, 10000, self.model.taskcla, t)
        loader = DataLoader(ds, batch_size=64, num_workers=0)
        for epoch in range(20):
            losses, hits = [], []
            for input, target in loader:
                input, target = input.to(self.device), target.to(self.device)
                bsz = input.shape[0]
                optimizer.zero_grad()
                out = self.model.head(input)
                loss = torch.nn.functional.cross_entropy(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss * bsz))
                TP = torch.sum(torch.argmax(out, dim=1) == target)
                hits.append(int(TP))
                scheduler.step_iter()
            scheduler.step_epoch()
            print(f"Epoch: {epoch}")
            print(f"Loss:{sum(losses) / len(ds):.2f}, Acc: {sum(hits) / len(ds):.2f}")

        self.model.head.eval()

    def train_backbone(self, t, trn_loader, val_loader):
        model = self.model
        optimizer, lr_scheduler = self._get_optimizer()
        best_loss, best_epoch, best_model = 1e8, 0, None
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(images)
                loss = self.criterion(t, out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                lr_scheduler.step_iter()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))

                train_loss.append(float(bsz * loss))
            lr_scheduler.step_epoch()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
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

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_model = copy.deepcopy(model)

            if epoch - best_epoch >= self.patience:
                break

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        print(f"Best epoch: {best_epoch}")
        self.model = best_model
        self.model.bb.fc = nn.Identity()
        torch.save(self.model.bb.state_dict(), "best.pth")

    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        eps = 1e-8
        self.model.eval()
        with torch.no_grad():
            classes = self.model.taskcla[t][1]
            self.model.task_offset.append(self.model.task_offset[-1] + classes)
            transforms = Compose([t for t in val_loader.dataset.transform.transforms
                                  if "CenterCrop" in t.__class__.__name__
                                  or "ToTensor" in t.__class__.__name__
                                  or "Normalize" in t.__class__.__name__])
            for c in range(classes):
                c = c + self.model.task_offset[t]
                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                val_indices = torch.tensor(val_loader.dataset.labels) == c
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    val_images = list(compress(val_loader.dataset.images, val_indices))
                    ds = ClassDirectoryDataset(train_images + val_images, transforms)
                else:
                    ds = np.concatenate((trn_loader.dataset.images[train_indices], val_loader.dataset.images[val_indices]), axis=0)
                    ds = ClassMemoryDataset(ds, transforms)
                loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=0, shuffle=False)
                from_ = 0
                class_features = torch.full((2 * len(ds), self.model.num_features), fill_value=-999999999.0, device=self.model.device)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    _, features = self.model(images, return_features=True)
                    class_features[from_: from_+bsz] = features
                    _, features = self.model(torch.flip(images, dims=(3,)), return_features=True)
                    class_features[from_+bsz: from_+2*bsz] = features
                    from_ += 2*bsz

                if self.remove_outliers:
                    median = torch.median(class_features, dim=0)[0]
                    dist = torch.cdist(class_features, median.unsqueeze(0), p=2).squeeze(1)
                    not_outliers = torch.topk(dist, int(0.99*class_features.shape[0]), largest=False, sorted=False)[1]
                    class_features = class_features[not_outliers]

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"
                is_ok = False
                while not is_ok:
                    try:
                        gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(self.device)
                        gmm.fit(class_features, delta=1e-3, n_iter=100)
                    except RuntimeError:
                        eps = 10 * eps
                        print(f"WARNING: Covariance matrix is singular. Compensation initialized. Changing eps to: {eps:.8f}")
                    else:
                        is_ok = True

                self.task_distributions.append(gmm)

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                targets = targets.to(self.device)
                # Forward current model
                _, features = self.model(images.to(self.device), return_features=True)
                hits_taw, hits_tag = self.calculate_metrics(features, targets, t)
                # Log
                total_loss = 0
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)

    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""

        # Task-Aware
        classes = self.model.task_offset[t+1] - self.model.task_offset[t]
        log_probs = [self.task_distributions[t].score_samples(features) for
                     t in range(self.model.task_offset[t], self.model.task_offset[t] + classes)]
        log_probs = torch.stack(log_probs, dim=1)
        class_id = torch.argmax(log_probs, dim=1) + self.model.task_offset[t]
        hits_taw = (class_id == targets).float()

        # Task-Agnostic
        pred = self.predict_class(features)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def predict_class(self, features):
        if self.use_head:
            return self.predict_class_head(features)
        return self.predict_class_bayes(features)

    def predict_class_bayes(self, features):
        with torch.no_grad():
            log_probs = [self.task_distributions[t].score_samples(features) for t in range(len(self.task_distributions))]
            log_probs = torch.stack(log_probs, dim=1)
            class_id = torch.argmax(log_probs, dim=1)
        return class_id

    def predict_class_head(self, features):
        with torch.no_grad():
            x = self.model.head(features)
            class_id = torch.argmax(x, dim=1)
        return class_id

    def _get_optimizer(self):
        """Returns the optimizer"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = WarmUpScheduler(optimizer, 100, 0.96)
        return optimizer, scheduler

