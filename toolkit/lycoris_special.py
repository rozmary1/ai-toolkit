import importlib
import math
import os
import weakref
from typing import Dict, Optional, Union, List, Type

import torch
from lycoris.kohya import (
    DyLoraModule,
    IA3Module,
    LoConModule,
    LohaModule,
    LokrModule,
    LycorisNetwork,
)
from lycoris.modules.glora import GLoRAModule
from torch import nn
from transformers import CLIPTextModel
from torch.nn import functional as F
from toolkit.network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]


def create_toolkit_module(base_cls: Type[nn.Module]) -> Type[nn.Module]:
    class ToolkitLycoModule(ToolkitModuleMixin, base_cls, ExtractableModuleMixin):
        def __init__(self, *args, network=None, **kwargs):
            ToolkitModuleMixin.__init__(self, network=network)
            base_cls.__init__(self, *args, **kwargs)

            # Ensure Toolkit forward wrapper always has the original forward available.
            # Some LyCORIS modules (e.g., LoHA) don't set `org_forward` themselves.
            if not hasattr(self, "org_forward"):
                org_module = getattr(self, "org_module", None)
                if isinstance(org_module, list) and org_module:
                    org_module = org_module[0]
                if org_module is not None and hasattr(org_module, "forward"):
                    self.org_forward = org_module.forward

    ToolkitLycoModule.__name__ = f"Toolkit{base_cls.__name__}"
    return ToolkitLycoModule


LohaSpecialModule = create_toolkit_module(LohaModule)
LokrSpecialModule = create_toolkit_module(LokrModule)
IA3SpecialModule = create_toolkit_module(IA3Module)
DyLoRASpecialModule = create_toolkit_module(DyLoraModule)
GLoRASpecialModule = create_toolkit_module(GLoRAModule)


def _optional_algorithm(module_path: str, class_name: str):
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        return None
    module = importlib.import_module(module_path)
    return getattr(module, class_name, None)


ButterflyOFTModule = _optional_algorithm("lycoris.modules.boft", "ButterflyOFTModule")
DiagOFTModule = _optional_algorithm("lycoris.modules.diag_oft", "DiagOFTModule")
FullModule = _optional_algorithm("lycoris.modules.full", "FullModule")

OPTIONAL_ALGO_MAP: Dict[str, Type[nn.Module]] = {}

if ButterflyOFTModule is not None:
    boft_cls = create_toolkit_module(ButterflyOFTModule)
    OPTIONAL_ALGO_MAP.update({
        "boft": boft_cls,
        "butterflyoft": boft_cls,
    })

if DiagOFTModule is not None:
    diag_oft_cls = create_toolkit_module(DiagOFTModule)
    OPTIONAL_ALGO_MAP.update({
        "diag-oft": diag_oft_cls,
        "diag_oft": diag_oft_cls,
        "diagoft": diag_oft_cls,
    })

if FullModule is not None:
    full_cls = create_toolkit_module(FullModule)
    OPTIONAL_ALGO_MAP.update({
        "full": full_cls,
    })


class LoConSpecialModule(ToolkitModuleMixin, LoConModule, ExtractableModuleMixin):
    def __init__(
            self,
            lora_name, org_module: nn.Module,
            multiplier=1.0,
            lora_dim=4, alpha=1,
            dropout=0., rank_dropout=0., module_dropout=0.,
            use_cp=False,
            network: 'LycorisSpecialNetwork' = None,
            use_bias=False,
            **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        # call super of super
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False

        # check if parent has bias. if not force use_bias to False
        if org_module.bias is None:
            use_bias = False

        self.scalar = nn.Parameter(torch.tensor(0.0))
        orig_module_name = org_module.__class__.__name__
        if orig_module_name in CONV_MODULES:
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.down_op = F.conv2d
            self.up_op = F.conv2d
            if use_cp and k_size != (1, 1):
                self.lora_down = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
                self.lora_mid = nn.Conv2d(lora_dim, lora_dim, k_size, stride, padding, bias=False)
                self.cp = True
            else:
                self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=use_bias)
        elif orig_module_name in LINEAR_MODULES:
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            if orig_module_name == 'GroupNorm':
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (56320x120 and 320x32)
                in_dim = org_module.num_channels
                out_dim = org_module.num_channels
            else:
                in_dim = org_module.in_features
                out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=use_bias)
        else:
            raise NotImplementedError
        self.shape = org_module.weight.shape

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lora_up.weight)
        if self.cp:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.register_load_state_dict_post_hook(self.load_weight_hook)

    def load_weight_hook(self, *args, **kwargs):
        self.scalar = nn.Parameter(torch.ones_like(self.scalar))


class LycorisSpecialNetwork(ToolkitNetworkMixin, LycorisNetwork):
    ALGO_MAP: Dict[str, Type[nn.Module]] = {
        "locon": LoConSpecialModule,
        "lycoris": LoConSpecialModule,
        "loha": LohaSpecialModule,
        "lokr": LokrSpecialModule,
        "ia3": IA3SpecialModule,
        "dylora": DyLoRASpecialModule,
        "glora": GLoRASpecialModule,
        **OPTIONAL_ALGO_MAP,
    }
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
        # 'UNet2DConditionModel',
        # 'Conv2d',
        # 'Timesteps',
        # 'TimestepEmbedding',
        # 'Linear',
        # 'SiLU',
        # 'ModuleList',
        # 'DownBlock2D',
        # 'ResnetBlock2D',  # need
        # 'GroupNorm',
        # 'LoRACompatibleConv',
        # 'LoRACompatibleLinear',
        # 'Dropout',
        # 'CrossAttnDownBlock2D', # needed
        # 'Transformer2DModel',  # maybe not, has duplicates
        # 'BasicTransformerBlock', # duplicates
        # 'LayerNorm',
        # 'Attention',
        # 'FeedForward',
        # 'GEGLU',
        # 'UpBlock2D',
        # 'UNetMidBlock2DCrossAttn'
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    def __init__(
            self,
            text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
            unet,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: float = 1,
            dropout: Optional[float] = None,
            rank_dropout: Optional[float] = None,
            module_dropout: Optional[float] = None,
            conv_lora_dim: Optional[int] = None,
            conv_alpha: Optional[float] = None,
            use_cp: Optional[bool] = False,
            network_module: Type[object] = LoConSpecialModule,
            train_unet: bool = True,
            train_text_encoder: bool = True,
            use_text_encoder_1: bool = True,
            use_text_encoder_2: bool = True,
            use_bias: bool = False,
            is_lorm: bool = False,
            peft_format: bool = False,
            base_model=None,
            **kwargs,
    ) -> None:
        # call ToolkitNetworkMixin super
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_lorm=is_lorm,
            **kwargs
        )
        # call the parent of the parent LycorisNetwork
        torch.nn.Module.__init__(self)

        # LyCORIS unique stuff
        algo = kwargs.pop("algo", None)
        module_algo_map = kwargs.pop("module_algo_map", None)
        name_algo_map = kwargs.pop("name_algo_map", None)
        target_replace_modules = kwargs.pop("target_lin_modules", None)
        target_replace_names = kwargs.pop("target_replace_names", None)

        if algo is not None:
            algo_cls = self.ALGO_MAP.get(algo.lower())
            if algo_cls is None:
                raise ValueError(f"Unknown LyCORIS algorithm: {algo}")
            network_module = algo_cls

        # keep a readable type label for saving/loading flows that expect it
        # (e.g., DoRA keymap adjustments in ToolkitNetworkMixin)
        self.network_type = algo or network_module.__name__
        self.peft_format = peft_format

        def _build_algo_map(raw_map: Optional[Dict[str, str]]):
            resolved = {}
            if raw_map is None:
                return resolved
            for key, value in raw_map.items():
                algo_cls = self.ALGO_MAP.get(value.lower())
                if algo_cls is None:
                    raise ValueError(f"Unknown LyCORIS algorithm for {key}: {value}")
                resolved[key] = algo_cls
            return resolved

        self.MODULE_ALGO_MAP = _build_algo_map(module_algo_map)
        self.NAME_ALGO_MAP = _build_algo_map(name_algo_map)

        if dropout is None:
            dropout = 0
        if rank_dropout is None:
            rank_dropout = 0
        if module_dropout is None:
            module_dropout = 0
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder
        self.base_model_ref = None
        if base_model is not None:
            self.base_model_ref = weakref.ref(base_model)

        self.torch_multiplier = None
        # triggers a tensor update
        self.multiplier = multiplier
        self.lora_dim = lora_dim

        if not self.ENABLE_CONV or conv_lora_dim is None:
            conv_lora_dim = 0
            conv_alpha = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            print('Apply different lora dim for conv layer')
            print(f'Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}')
        elif self.conv_lora_dim == 0:
            print('Disable conv layer')

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            print('Apply different alpha value for conv layer')
            print(f'Conv alpha: {conv_alpha}, Linear alpha: {alpha}')

        if 1 >= dropout >= 0:
            print(f'Use Dropout value: {dropout}')
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # create module instances
        unet_target_modules = (
            target_replace_modules
            if target_replace_modules is not None
            else list(LycorisSpecialNetwork.UNET_TARGET_REPLACE_MODULE)
        )
        unet_target_names = (
            target_replace_names
            if target_replace_names is not None
            else list(LycorisSpecialNetwork.UNET_TARGET_REPLACE_NAME)
        )

        if kwargs.get("is_transformer"):
            for transformer_module in ["Transformer2DModel", "ZImageTransformer2DModel"]:
                if transformer_module not in unet_target_modules:
                    unet_target_modules.append(transformer_module)

        def create_modules(
                prefix,
                root_module: torch.nn.Module,
                target_replace_modules,
                target_replace_names=[]
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            # remove this
            named_modules = root_module.named_modules()
            # add a few to tthe generator

            for name, module in named_modules:
                module_name = module.__class__.__name__
                if module_name in target_replace_modules:
                    if module_name in self.MODULE_ALGO_MAP:
                        algo = self.MODULE_ALGO_MAP[module_name]
                    else:
                        algo = network_module
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        if lora_name.startswith('lora_unet_input_blocks_1_0_emb_layers_1'):
                            print(f"{lora_name}")

                        if child_module.__class__.__name__ in LINEAR_MODULES and lora_dim > 0:
                            lora = algo(
                                lora_name, child_module, self.multiplier,
                                self.lora_dim, self.alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        elif child_module.__class__.__name__ in CONV_MODULES:
                            k_size, *_ = child_module.kernel_size
                            if k_size == 1 and lora_dim > 0:
                                lora = algo(
                                    lora_name, child_module, self.multiplier,
                                    self.lora_dim, self.alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    network=self,
                                    parent=module,
                                use_bias=use_bias,
                                    **kwargs
                                )
                            elif conv_lora_dim > 0:
                                lora = algo(
                                    lora_name, child_module, self.multiplier,
                                    self.conv_lora_dim, self.conv_alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    network=self,
                                    parent=module,
                                    use_bias=use_bias,
                                    **kwargs
                                )
                            else:
                                continue
                        else:
                            continue
                        loras.append(lora)
                elif name in target_replace_names:
                    if name in self.NAME_ALGO_MAP:
                        algo = self.NAME_ALGO_MAP[name]
                    else:
                        algo = network_module
                    lora_name = prefix + '.' + name
                    if module.__class__.__name__ == 'Linear' and lora_dim > 0:
                        lora = algo(
                            lora_name, module, self.multiplier,
                            self.lora_dim, self.alpha,
                            self.dropout, self.rank_dropout, self.module_dropout,
                            use_cp,
                            parent=module,
                            network=self,
                            use_bias=use_bias,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                self.lora_dim, self.alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        elif conv_lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                self.conv_lora_dim, self.conv_alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)
            return loras

        if issubclass(network_module, GLoRAModule):
            print('GLoRA enabled, only train transformer')
            # only train transformer (for GLoRA)
            unet_target_modules = [
                "Transformer2DModel",
                "Attention",
            ]
            unet_target_names = []

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        if self.train_text_encoder:
            for i, te in enumerate(text_encoders):
                if not use_text_encoder_1 and i == 0:
                    continue
                if not use_text_encoder_2 and i == 1:
                    continue
                self.text_encoder_loras.extend(create_modules(
                    LycorisSpecialNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i + 1}' if use_index else ''),
                    te,
                    LycorisSpecialNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")
        if self.train_unet:
            self.unet_loras = create_modules(
                LycorisSpecialNetwork.LORA_PREFIX_UNET,
                unet,
                unet_target_modules,
                unet_target_names,
            )
        else:
            self.unet_loras = []
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
