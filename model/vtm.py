import torch.nn as nn
from .reshape import forward_6d_as_4d, from_6d_to_3d, from_3d_to_6d
from .attention import CrossAttention
        
        
class VTMImageBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def forward(self, x, t_idx=None, **kwargs):
        return forward_6d_as_4d(self.backbone, x, t_idx=t_idx, get_features=True, **kwargs)

    # 여기가 중요한 점이, 내가 prompt parameter 이런 방식으로 넘기면 될 것으로 보인다.
    def bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.backbone.bias_parameters():
            yield p

    def bias_parameter_names(self):
        return [f'backbone.{name}' for name in self.backbone.bias_parameter_names()]


class VTMLabelBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def encode(self, x, t_idx=None, **kwargs):
        return forward_6d_as_4d(self.backbone.encode, x, t_idx=t_idx, **kwargs)
    
    def decode(self, x, t_idx=None, **kwargs):
        return forward_6d_as_4d(self.backbone.decode, x, t_idx=t_idx, **kwargs)
    
    def forward(self, x, t_idx=None, encode_only=False, decode_only=False, **kwargs):
        assert not (encode_only and decode_only)
        if not decode_only:
            x = self.encode(x, t_idx=t_idx, **kwargs)
        if not encode_only:
            x = self.decode(x, t_idx=t_idx, **kwargs)
        
        return x
        
# todo 너무 구현이 간단하잖아! EXPRES 적용하기 매우 좋다.
"""
1. bias tuning on V
2. bias tuning on Q
3. VTM module에 nn parameter 느낌으로 prompt 넣기, VPT-deep 비슷하게
4. VTM module에서 prompt는 inter-layer connection 넣기 w/ residual connection
5. 

"""
class VTMMatchingModule(nn.Module):
    def __init__(self, dim_w, dim_z, config):
        super().__init__()
        self.matching = nn.ModuleList([CrossAttention(dim_w, dim_z, dim_z, num_heads=config.n_attn_heads)
                                       for i in range(config.n_levels)])
        self.n_levels = config.n_levels
            
    def forward(self, W_Qs, W_Ss, Z_Ss, attn_mask=None):
        B, T, N, _, H, W = W_Ss[-1].size()
        
        assert len(W_Qs) == self.n_levels
        
        if attn_mask is not None:
            attn_mask = from_6d_to_3d(attn_mask)
            
        Z_Qs = []
        for level in range(self.n_levels):
            Q = from_6d_to_3d(W_Qs[level])
            K = from_6d_to_3d(W_Ss[level])
            V = from_6d_to_3d(Z_Ss[level])

            # K에 대해서는 layer Norm이 적용되면 좋을 것으로 생각했는데, 그렇지 않은 것 같다.
            O = self.matching[level](Q, K, V, N=N, H=H, mask=attn_mask)
            Z_Q = from_3d_to_6d(O, B=B, T=T, H=H, W=W)
            Z_Qs.append(Z_Q)
        
        return Z_Qs


class VTM(nn.Module):
    def __init__(self, image_backbone, label_backbone, matching_module):
        super().__init__()
        self.image_backbone = image_backbone
        self.label_backbone = label_backbone
        self.matching_module = matching_module
        
        self.n_levels = self.image_backbone.backbone.n_levels

    def bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.image_backbone.bias_parameters():
            yield p

    def bias_parameter_names(self):
        return [f'image_backbone.{name}' for name in self.image_backbone.bias_parameter_names()]

    def pretrained_parameters(self):
        return self.image_backbone.parameters()
    
    def scratch_parameters(self):
        modules = [self.label_backbone, self.matching_module]
        for module in modules:
            for p in module.parameters():
                yield p
        
    def forward(self, X_S, Y_S, X_Q, t_idx=None, sigmoid=True):
        # encode support input, query input, and support output
        W_Ss = self.image_backbone(X_S, t_idx)
        W_Qs = self.image_backbone(X_Q, t_idx)
        Z_Ss = self.label_backbone.encode(Y_S, t_idx)
        
        # mix support output by matching
        Z_Q_preds = self.matching_module(W_Qs, W_Ss, Z_Ss)
        
        # decode support output
        Y_Q_pred = self.label_backbone.decode(Z_Q_preds, t_idx)
        
        if sigmoid:
            Y_Q_pred = Y_Q_pred.sigmoid()
        
        return Y_Q_pred
