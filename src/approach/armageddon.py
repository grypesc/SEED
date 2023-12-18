import copy
import os
import random
from itertools import compress

import numpy as np
import torch

from argparse import ArgumentParser
from torch import nn
from PIL import Image

from .incremental_learning import Inc_Learning_Appr
from src.networks.resnet32 import resnet20, resnet32
from .mvgb import ClassDirectoryDataset, ClassMemoryDataset
from torchvision.transforms import ToPILImage, Compose

torch.backends.cuda.matmul.allow_tf32 = False
# export PYTHONPATH=~/facil


def batch_to_numpy_images(images_batch):
    images = images_batch.cpu().permute(0, 2, 3, 1)
    mean = torch.tensor([0.5071, 0.4866, 0.4409]).unsqueeze(0).unsqueeze(0)
    std = torch.tensor([0.2009, 0.1984, 0.2023]).unsqueeze(0).unsqueeze(0)
    return np.array(torch.clip(255 * (images * std + mean), min=0, max=255), dtype=np.uint8)


class MembeddingDataset(torch.utils.data.Dataset):
    def __init__(self, membeddings_per_class: int):
        self.labels = torch.zeros((0,), dtype=torch.int64)
        self.membeddings = torch.zeros((0, 512), dtype=torch.float)
        self.membeddings_per_class = membeddings_per_class
        self.reconstructed = np.zeros((0, 32, 32, 3), dtype=np.uint8)
        self.images = np.zeros((0, 32, 32, 3), dtype=np.uint8)
        self.transforms = None

    def __len__(self):
        return self.membeddings.shape[0]

    def __getitem__(self, index):
        reconstructed = self.transforms(self.reconstructed[index])
        return self.membeddings[index], reconstructed, self.labels[index]

    def set_transforms(self, transforms):
        self.transforms = Compose((ToPILImage(), transforms))

    def add(self, label, new_membeddings, new_reconstructed, new_images):
        new_labels = label.expand(new_membeddings.shape[0])
        self.labels = torch.cat((self.labels, new_labels), dim=0)
        self.membeddings = torch.cat((self.membeddings, new_membeddings), dim=0)

        new_reconstructed = batch_to_numpy_images(new_reconstructed)
        self.reconstructed = np.concatenate((self.reconstructed, new_reconstructed), axis=0)

        new_images = batch_to_numpy_images(new_images)
        self.images = np.concatenate((self.images, new_images), axis=0)


class SlowLearner(nn.Module):

    class EncoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            planes = 32
            self.layers = nn.Sequential(nn.Conv2d(3, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(2 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(4 * planes, 8 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(8 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(4 * planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(2 * planes, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                        nn.Conv2d(planes, 8, kernel_size=3, stride=1, padding=1)
                                        )

        def forward(self, x):
            x = self.layers(x)
            return x

    class DecoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            planes = 32
            self.layers1 = nn.Sequential(nn.Conv2d(8, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2),
                                         nn.Conv2d(2 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(4 * planes, 8 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2)
                                         )

            self.layers2 = nn.Sequential(nn.Conv2d(8 * planes, 4 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(4 * planes, 2 * planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(2 * planes, planes, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                         nn.Conv2d(planes, 3, kernel_size=3, stride=1, padding=1)
                                         )

        def forward(self, z):
            x = z.reshape(z.shape[0], 8, 8, 8)
            feature_maps = self.layers1(x)
            return feature_maps, self.layers2(feature_maps)

    def __init__(self, z_size):
        super().__init__()
        self.z_size = z_size
        self.encoder = SlowLearner.EncoderBlock()
        self.decoder = SlowLearner.DecoderBlock()

    def forward(self, x, decode=True):
        x = self.encoder(x)
        z = x.reshape(x.shape[0], -1)
        if not decode:
            return z
        feature_maps, x = self.decoder(x)
        return z, feature_maps, x


class Adaptator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 128, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()
        self.out = nn.Conv2d(128, 8, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        return self.out(x)


class Appr(Inc_Learning_Appr):
    """https://www.youtube.com/watch?v=wfa9xH3cJ8E"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, membeddings=100, slow_epochs=200, fast_epochs=200, slow_lr=1e-3, fast_lr=1e-3, slow_wd=1e-8,
                 fast_wd=1e-5, freeze_encoder=False, adapt_membeddings=False, alpha=0.5):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.task_offset = [0]
        self.model = None
        self.membeddings_per_class = membeddings
        self.membeddings_per_class_val = 100
        self.mem_train_dataset = MembeddingDataset(self.membeddings_per_class)
        self.mem_valid_dataset = MembeddingDataset(self.membeddings_per_class_val)
        self.slow_epochs = slow_epochs
        self.fast_epochs = fast_epochs
        self.slow_lr = slow_lr
        self.fast_lr = fast_lr
        self.slow_wd = slow_wd
        self.fast_wd = fast_wd
        self.freeze_encoder = freeze_encoder
        self.adapt_membeddings = adapt_membeddings
        self.alpha = alpha

        self.slow_learner = SlowLearner(512)
        self.slow_learner.to(device)
        self.fast_learner = resnet32()
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--membeddings',
                            help='number of memory embeddings per class',
                            type=int,
                            default=100)
        parser.add_argument('--slow-epochs',
                            help='epochs of slow learner',
                            type=int,
                            default=200)
        parser.add_argument('--fast-epochs',
                            help='epochs of fast learner',
                            type=int,
                            default=200)
        parser.add_argument('--slow-lr',
                            help='learning rate of slow learner',
                            type=float,
                            default=1e-3)
        parser.add_argument('--slow-wd',
                            help='weight decay of slow learner',
                            type=float,
                            default=1e-4)
        parser.add_argument('--fast-lr',
                            help='learning rate of fast learner',
                            type=float,
                            default=1e-1)
        parser.add_argument('--fast-wd',
                            help='weight decay of fast learner',
                            type=float,
                            default=1e-4)
        parser.add_argument('--freeze-encoder',
                            help='freeze encoder after first task',
                            action='store_true',
                            default=False)
        parser.add_argument('--adapt-membeddings',
                            help='use MLP to update membeddings in memory using features adaptation',
                            action='store_true',
                            default=False)
        parser.add_argument('--alpha',
                            help='alpha',
                            type=float,
                            default=0.5)

        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        old_slow_learner = copy.deepcopy(self.slow_learner)
        if not self.freeze_encoder or t == 0:
            print(f"Training slow learner on task {t}")
            self.train_slow_learner(trn_loader, val_loader)
        # state_dict = torch.load("slow_learner_10.pth")
        # self.slow_learner.load_state_dict(state_dict, strict=True)
        self.manage_memory(old_slow_learner, t, trn_loader, val_loader, val_loader.dataset.transform)
        print(f"Training fast learner after task {t}")
        self.train_fast_learner(t, trn_loader, val_loader)
        self.dump_visualizations(t)

    def train_slow_learner(self, trn_loader, val_loader):
        model = self.slow_learner
        model.to(self.device)
        print(f'Slow learner has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        epochs = self.slow_epochs
        milestones = [50, 100, 150, 190]
        if len(self.mem_train_dataset) > 0:
            self.mem_train_dataset.set_transforms(trn_loader.dataset.transform)
            self.mem_valid_dataset.set_transforms(val_loader.dataset.transform)
            mem_train_loader = torch.utils.data.DataLoader(self.mem_train_dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True)
            mem_val_loader = torch.utils.data.DataLoader(self.mem_valid_dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True)
            epochs = self.slow_epochs // 2
            milestones = [40, 60, 80, 95]
        optimizer, lr_scheduler = self._get_slow_optimizer(model, self.slow_wd, milestones=milestones)
        for epoch in range(epochs):
            train_loss, valid_loss = [], []
            model.train()
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                optimizer.zero_grad()
                _, _, reconstructed = model(images)
                loss = nn.functional.mse_loss(reconstructed, images)

                if len(self.mem_train_dataset) > 0 and self.alpha > 0.0:
                    mem_loader_iter = iter(mem_train_loader)
                    mem_images = next(mem_loader_iter)[1].to(self.device)
                    _, _, mem_reconstructed = model(mem_images)
                    mem_loss = nn.functional.mse_loss(mem_reconstructed, mem_images)
                    loss = (1 - self.alpha) * loss + self.alpha * mem_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    z, _, reconstructed = model(images)
                    loss = nn.functional.mse_loss(reconstructed, images)

                    if len(self.mem_train_dataset) > 0 and self.alpha > 0.0:
                        mem_loader_iter = iter(mem_val_loader)
                        mem_images = next(mem_loader_iter)[1].to(self.device)
                        _, _, mem_reconstructed = model(mem_images)
                        mem_loss = nn.functional.mse_loss(mem_reconstructed, mem_images)
                        loss = (1 - self.alpha) * loss + self.alpha * mem_loss

                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {100*train_loss:.2f} Val loss: {100*valid_loss:.2f}")
        self.slow_learner = model
        torch.save(self.slow_learner.state_dict(), f"slow_learner.pth")

    def train_fast_learner(self, t, trn_loader, val_loader):
        model = resnet32()
        model.fc = nn.Linear(64, self.task_offset[t+1])
        model.to(self.device)
        print(f'Fast learner has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        self.mem_train_dataset.set_transforms(trn_loader.dataset.transform)
        self.mem_valid_dataset.set_transforms(val_loader.dataset.transform)
        mem_train_loader = torch.utils.data.DataLoader(self.mem_train_dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True)
        mem_val_loader = torch.utils.data.DataLoader(self.mem_valid_dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True)

        optimizer, lr_scheduler = self._get_fast_optimizer(model, self.fast_wd)
        for epoch in range(self.fast_epochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for _, reconstructed, targets in mem_train_loader:
                bsz = reconstructed.shape[0]
                reconstructed, targets = reconstructed.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(reconstructed)
                loss = self.criterion(out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for _, reconstructed, targets in mem_val_loader:
                    bsz = reconstructed.shape[0]
                    reconstructed, targets = reconstructed.to(self.device), targets.to(self.device)
                    out = model(reconstructed)
                    loss = self.criterion(out, targets)
                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(mem_train_loader.dataset)
            valid_loss = sum(valid_loss) / len(mem_val_loader.dataset)
            train_acc = train_hits / len(mem_train_loader.dataset)
            val_acc = val_hits / len(mem_val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")
        self.fast_learner = model

    def membeddings_adaptation(self, old_slow_learner, trn_loader, val_loader):
        print("Features adaptation:")
        old_slow_learner.eval()
        self.slow_learner.eval()
        model = Adaptator()
        model.to(self.device)
        # Train feature adaptator network
        optimizer, lr_scheduler = self._get_fast_optimizer(model, self.slow_wd, milestones=[30, 60, 90])
        for epoch in range(self.slow_epochs // 2):
            train_loss, valid_loss = [], []
            model.train()
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    old_features = old_slow_learner.encoder(images)
                    new_features = self.slow_learner.encoder(images)
                estimated_new_features = model(old_features)
                loss = nn.functional.mse_loss(estimated_new_features, new_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    old_features = old_slow_learner.encoder(images)
                    new_features = self.slow_learner.encoder(images)
                    estimated_new_features = model(old_features)
                    loss = nn.functional.mse_loss(estimated_new_features, new_features)
                    valid_loss.append(float(bsz * loss))
            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {100*train_loss:.2f} Val loss: {100*valid_loss:.2f}")

        # Adapt features
        with torch.no_grad():
            for mem_dataset in [self.mem_train_dataset, self.mem_valid_dataset]:
                mem_loader = torch.utils.data.DataLoader(mem_dataset, batch_size=128, num_workers=0, shuffle=False)
                index = 0
                for old_membedding, _, _ in mem_loader:
                    bsz = old_membedding.shape[0]
                    old_membedding = old_membedding.to(self.device).reshape(bsz, 8, 8, 8)
                    # _, new_reconstructed = self.slow_learner.decoder(old_membedding)
                    new_membedding = model(old_membedding)
                    new_reconstructed = self.slow_learner.decoder(new_membedding)[1]
                    new_membedding = new_membedding.reshape(bsz, -1)
                    mem_dataset.membeddings[index:index + bsz] = new_membedding.cpu()
                    mem_dataset.reconstructed[index:index + bsz] = batch_to_numpy_images(new_reconstructed.cpu())
                    index += bsz

    def manage_memory(self, old_slow_learner, t, trn_loader, val_loader, transforms):
        self.mem_train_dataset.set_transforms(transforms)
        self.mem_valid_dataset.set_transforms(transforms)

        # Update old reconstruction
        if len(self.mem_train_dataset) > 0:
            # self.update_memory(self.mem_train_dataset)
            # self.update_memory(self.mem_valid_dataset)
            self.membeddings_adaptation(old_slow_learner, trn_loader, val_loader)

        classes_ = set(trn_loader.dataset.labels)
        self.task_offset += [len(classes_) + self.task_offset[t]]

        # Add new train membeddings to memory
        self.add_membeddings_to_memory(self.mem_train_dataset, trn_loader.dataset, t, transforms, self.membeddings_per_class)
        self.add_membeddings_to_memory(self.mem_valid_dataset, val_loader.dataset, t, transforms, self.membeddings_per_class_val)

    @torch.no_grad()
    def update_memory(self, mem_dataset):
        self.slow_learner.eval()
        mem_loader = torch.utils.data.DataLoader(mem_dataset, batch_size=128, num_workers=0, shuffle=False)
        index = 0
        for old_membedding, old_reconstructed, _ in mem_loader:
            bsz = old_membedding.shape[0]
            old_membedding, old_reconstructed = old_membedding.to(self.device), old_reconstructed.to(self.device)
            # _, new_reconstructed = self.slow_learner.decoder(old_membedding)
            new_membedding, _, new_reconstructed = self.slow_learner(old_reconstructed, decode=True)
            mem_dataset.membeddings[index:index + bsz] = new_membedding.cpu()
            mem_dataset.reconstructed[index:index + bsz] = batch_to_numpy_images(new_reconstructed.cpu())
            index += bsz

    @torch.no_grad()
    def add_membeddings_to_memory(self, mem_dataset, src_dataset, t, transforms, num_to_store):
        self.slow_learner.eval()
        labels = np.array(src_dataset.labels)
        classes_ = set(src_dataset.labels)
        for i in classes_:
            class_indices = labels == i
            if isinstance(src_dataset.images, list):
                train_images = list(compress(src_dataset.images, class_indices))
                train_images = train_images[:num_to_store]
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = src_dataset.images[class_indices][:num_to_store]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=self.membeddings_per_class, num_workers=0, shuffle=True)
            for images in loader:
                images = images.to(self.device)
                membeddings, _, reconstructed = self.slow_learner(images, decode=True)
                membeddings, reconstructed = membeddings.cpu(), reconstructed.cpu()
                mem_dataset.add(torch.tensor(i), membeddings, reconstructed, images.cpu())

    def dump_visualizations(self, t):
        visualizations_dir = f"{self.logger.exp_path}/visualizations_{str(self.logger.begin_time.strftime('%Y-%m-%d_%H:%M:%S'))}/{t}"
        os.makedirs(visualizations_dir, exist_ok=True)
        classes = np.unique(self.mem_valid_dataset.labels)
        for c in classes:
            index = np.argmax(self.mem_valid_dataset.labels == c)
            reconstructed = self.mem_valid_dataset.reconstructed[index]
            reconstructed = Image.fromarray(reconstructed)
            reconstructed.save(f"{visualizations_dir}/{c}_membedding.png")
            image = self.mem_valid_dataset.images[index]
            image = Image.fromarray(image)
            image.save(f"{visualizations_dir}/{c}_image.png")

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.slow_learner.eval()
        self.fast_learner.eval()
        for images, targets in val_loader:
            targets = targets.to(self.device)
            # Forward current model
            _, _, reconstructed = self.slow_learner(images.to(self.device), decode=True)
            logits = self.fast_learner(reconstructed)
            preds = torch.argmax(logits, dim=1)
            hits_tag = preds == targets
            preds = torch.argmax(logits[:, self.task_offset[t]:self.task_offset[t+1]], dim=1) + self.task_offset[t]
            hits_taw = preds == targets
            # Log
            total_loss = 0
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def _get_slow_optimizer(self, model, wd, milestones=[60, 120, 160]):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.slow_lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def _get_fast_optimizer(self, model, wd, milestones=[60, 120, 160]):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.fast_lr, weight_decay=wd, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler
