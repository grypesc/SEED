import torch
from torch import nn
from copy import deepcopy

from .resnet32_linear_turbo import resnet32
from .resnet_linear_turbo import resnet18, resnet34, resnet50
from .resnet32_linear_bottleneck import resnet20


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, taskcla, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        self.taskcla = taskcla
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass


class ExtractorEnsemble(LLL_Net):

    def __init__(self, backbone, taskcla, network_type, device):
        super().__init__(backbone, taskcla, remove_existing_head=False)
        self.model = None
        self.num_features = 64
        self.network_type = network_type
        if network_type == "resnet18":
            self.bb_fun = resnet18
        elif network_type == "resnet34":
            self.bb_fun = resnet34
        elif network_type == "resnet50":
            self.bb_fun = resnet50
        elif network_type == "resnet32":
            self.bb_fun = resnet32
        elif network_type == "resnet20":
            self.num_features = 24
            self.bb_fun = resnet20
        else:
            raise RuntimeError("Network not supported")

        self.bbs = nn.ModuleList([])
        self.head = nn.Identity()

        # Uncomment to load a model, set 6 to number of experts that's in .pth, comment backbone training
        # self.bbs = nn.ModuleList([copy.deepcopy(bb) for _ in range(min(len(taskcla), 6))])
        # for bb in self.bbs:
        #     bb.fc = nn.Identity()
        # state_dict = torch.load("seb-resnet32.pth")
        # self.load_state_dict(state_dict, strict=True)

        self.task_offset = [0]
        self.taskcla = taskcla
        self.device = device

    def add_head(self, num_outputs):
        pass

    def forward(self, x):
        # semi_features = self.bbs[0].calculate_semi_features(x)
        features = [bb.forward(x) for bb in self.bbs]
        return torch.stack(features, dim=1)

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        pass
