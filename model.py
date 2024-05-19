import torch
from torch import nn
from torchvision.models import resnet, efficientnet_b0
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig
from bottleneck_transformer_pytorch import BottleStack

# ResNet50 Regression Model
class RegressionModel(nn.Module):

    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(RegressionModel, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=2048, out_features=256, bias=True),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x
    
# bottlenenck transfmormer layer 
layer = BottleStack(
    dim = 2048,
    fmap_size = 7, # set specifically for imagenet's 224 x 224
    dim_out = 2048,
    proj_factor = 4,
    downsample = False,
    heads = 4,
    dim_head = 128,
    rel_pos_emb = True,
    activation = nn.ReLU()
)  

# Lightweight Transformer - ResNet50 Model
class RegressionModel_Transformer(nn.Module):

    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(RegressionModel_Transformer, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights
        

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            # Remove the last layer of the model Res
            backbone = list(self.model.children())
            self.model = nn.Sequential(*backbone[:-2])
            #transformer layer
            self.layer = layer 
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten(1)
            self.linear = nn.Linear(2048, self.output_nodes,bias=True)
            
            
    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


# Set for ResNet Model
model_res = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_res = model_res.to(device)

def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

# Remove the last layer of the model Res
layers_Res = list(model_res.children())
model_Res = nn.Sequential(*layers_Res[:-1])


# Set the top layers to be not trainable
count = 0
for child in model_Res.children():
    count += 1
    if count < 8:
        for param in child.parameters():
            param.requires_grad = False
 

# Set for ViT Model
model_trans = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
count = 0
for child in model_trans.children():
    count += 1
    if count < 4:
        for param in child.parameters():
            param.requires_grad = False

layers_trans = list(model_trans.children()) # Get all the layers from the Transformer model
model_trans_top = nn.Sequential(*layers_trans[:-2]) # Remove the normalization layer and pooler layer
trans_layer_norm = list(model_trans.children())[2] # Get the normalization layer

class model_vit(nn.Module):
    def __init__(self, model_trans_top, trans_layer_norm, model_Res, dp_rate = 0.3):
        super().__init__()
        # All the trans model layers
        self.model_trans_top = model_trans_top
        self.trans_layer_norm = trans_layer_norm
        self.trans_flatten = nn.Flatten()
        self.trans_linear = nn.Linear(150528, 2048)

        # All the ResNet model
        self.model_Res = model_Res

        # Merge the result and pass the
        self.dropout = nn.Dropout(dp_rate)
        self.linear1 = nn.Linear(2048, 500)# 4096
        self.linear2 = nn.Linear(500,1)

    def forward(self, trans_b,res_b):
        # Get intermediate outputs using hidden layer
        result_trans = self.model_trans_top(trans_b)
        patch_state = result_trans.last_hidden_state[:,1:,:] # Remove the classification token and get the last hidden state of all patchs
        result_trans = self.trans_layer_norm(patch_state)
        result_trans = self.trans_flatten(patch_state)
        result_trans = self.dropout(result_trans)
        result_trans = self.trans_linear(result_trans)

        result_res = self.model_Res(res_b)
        result_res = torch.reshape(result_res, (1, 2048))

        result_merge = torch.cat((result_trans, result_res),1)
        result_merge = result_trans
        result_merge = self.dropout(result_merge)
        result_merge = self.linear1(result_merge)
        result_merge = self.dropout(result_merge)
        result_merge = self.linear2(result_merge)

        return result_merge

model = model_vit(model_trans_top, trans_layer_norm, model_Res, dp_rate = 0.3)

print(num_trainable_params(model))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)