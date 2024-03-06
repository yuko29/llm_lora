# Code adapted from https://github.com/HanGuo97/lq-lora/blob/main/models/lora_utils.py
import os
import math
import torch
import click
from transformers import (
    GPTQConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    RobertaForSequenceClassification)
from transformers.trainer import Trainer
from torch.utils import _pytree as pytree
from peft.tuners import lora
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training)
from typing import List, Optional, Union, Dict, Any, cast

# from .factorizations_utils import vanilla_low_rank_decomposition, get_lra_error


def _enable_gradient_checkpointing(model: LlamaForCausalLM) -> None:
    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()


def prepare_model_for_lora(
    model: LlamaForCausalLM,
    num_ranks: int,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_preprocess_embedding: bool = False,
) -> PeftModelForCausalLM:

    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(f"Expected LlamaForCausalLM, but got {type(model)}")
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=num_ranks,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        # use_rslora=True,
        )
    
    
    # the inputs does not have requires_grad set to True, so add forward hook to make input require grad
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # if use_gradient_checkpointing is True:
    #     _enable_gradient_checkpointing(model)
        
    if checkpoint_dir is not None:
        click.secho(
            f"Loading PEFT model from {checkpoint_dir}. "
            f"Aforementioned arguments will be ignored",
            fg="blue")
        new_model = PeftModelForCausalLM.from_pretrained(model=model,
                                                        model_id=checkpoint_dir,
                                                        is_trainable=True)
        # TODO: Update checkpoint lora setting
    else:
        new_model = get_peft_model(model, peft_config)
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(new_model)}")
    return new_model


# def transform_lora_layers(
#     model: Union[PeftModelForCausalLM, PeftModelForSequenceClassification],
#     init_weights: Dict[str, torch.Tensor],
#     device: Optional[torch.device] = None,
# ) -> None:

#     click.secho(
#         f"Transforming LoRA layers\n",
#         fg="yellow")

#     if not isinstance(model, PeftModelForCausalLM):
#         raise NotImplementedError(f"Unknown model type: {type(model)}")
    

#     for name, submodule in model.base_model.model.named_modules():

#         # This implicitly assumes that `LoraLayer`
#         # do not include `LoraLayer` within the module.
#         if isinstance(submodule, lora.LoraLayer):

#             # These operations will be too slow on CPU
#             if device is not None:
#                 # This is in-place
#                 submodule.to(device=device)
            
#             # assert_lora_Linear_layer(submodule)
#             lra_error = init_weights[name] 
            
#             num_ranks = cast(
#                 int,
#                 submodule.r[submodule.active_adapter[0]])
            
#             print(f"{name:<50}")
#             transform_lora_layer(
#                 submodule,
#                 num_ranks=num_ranks,
#                 W=lra_error)


# @torch.no_grad()
# def transform_lora_layer(
#     module: lora.LoraLayer,
#     num_ranks: int,
#     W: Optional[torch.Tensor] = None,
# ) -> None:

#     if type(module) is lora.Linear:
#         L1, L2= vanilla_low_rank_decomposition(
#             W,
#             num_ranks=num_ranks,
#             log_level="None")
#     else:
#         raise TypeError

#     # The LoRA layer essentially does the following computation:
#     # ```
#     # 1. x_ = dropout(x)
#     # 2. y  = x @ W.T + s * x_ @ A.T @ B.T
#     # When dropout is turned off,
#     #    y = x @  W.T +  s * x @ A.T @          B.T
#     #      = x @ (W.T +  s *     A.T @          B.T)
#     #      = x @ (W   +  s *     B   @          A  ).T
#     #      = x @ (W   + [sqrt(s) B]  @ [sqrt(s) A] ).T
#     # ```
#     # Since LPQ applies the following computation: `W + L1 @ L2`, we want
#     # ```
#     # 1. L1 = sqrt(s) B
#     # 2. L2 = sqrt(s) A
#     # ```
#     # Hence we assign
#     # ```
#     # 1. A = L2 / sqrt(s)
#     # 2. B = L1 / sqrt(s)
#     # ```
#     scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
#     module.lora_A[module.active_adapter[0]].weight.copy_(L2 / scale_sqrt)
#     module.lora_B[module.active_adapter[0]].weight.copy_(L1 / scale_sqrt)


def assert_lora_Linear_layer(
    module: lora.Linear,
) -> None:
    if type(module) is not lora.Linear:
        raise TypeError
    if module.fan_in_fan_out:
        raise ValueError
    if (len(module.lora_embedding_A) != 0 or
        len(module.lora_embedding_B) != 0):
        raise ValueError
    if (module.bias is not None or
        module.lora_A[module.active_adapter[0]].bias is not None or
        module.lora_B[module.active_adapter[0]].bias is not None):
        raise ValueError

    lora_B_weight = module.lora_B[module.active_adapter[0]].weight
    lora_B_weight_all_zeros = (lora_B_weight == 0.).all().item()
    if not lora_B_weight_all_zeros:
        raise ValueError("Expected `module.lora_B.weight` to be zero.")


# Replace the base linear layer weight of LoRA layer with a new weight
# def replace_weight_(
#     module: Union[lora.Linear, torch.nn.Linear],
#     new_weight: Union[torch.Tensor, tensor_container_utils.QuantizedTensor],
# ) -> None:
#     if isinstance(new_weight, tensor_container_utils.QuantizedTensor):
#         if not isinstance(module.weight, torch.nn.Parameter):
#             raise TypeError
#         if module.weight.requires_grad is not False:
#             raise ValueError
#         module.weight = torch.nn.Parameter(
#             new_weight,
#             requires_grad=module.weight.requires_grad)
#     else:
#         module.weight.copy_(new_weight)

# def prepare_model_for_lora_classification(
#     model: RobertaForSequenceClassification,
#     num_ranks: int,
#     lora_alpha: int = 16,
#     lora_dropout: float = 0.05,
#     target_modules: Optional[List[str]] = None,
#     use_gradient_checkpointing: bool = False,
# ) -> PeftModelForSequenceClassification:

#     if not isinstance(model, RobertaForSequenceClassification):
#         raise TypeError(f"Expected RobertaForSequenceClassification, but got {type(model)}")
#     if target_modules is None:
#         target_modules = [
#             "query",
#             "key",
#             "value",
#             "output.dense",
#             "intermediate.dense",
#         ]

#     click.secho(
#         f"Applying LoRA with the following configurations:\n"
#         f"\t -num_ranks: {num_ranks}\n"
#         f"\t -lora_alpha: {lora_alpha}\n"
#         f"\t -lora_dropout: {lora_dropout}\n"
#         f"\t -target_modules: {target_modules}",
#         fg="blue")

#     peft_config = LoraConfig(
#         r=num_ranks,
#         lora_alpha=lora_alpha,
#         target_modules=target_modules,
#         lora_dropout=lora_dropout,
#         bias="none",
#         task_type="SEQ_CLS")

#     # This function is useful even if we are not doing int8 training.
#     # (1) Freezing all the parameters before we add LoRA.
#     # (2) Casting all fp16/bf16 parameters to fp32.
#     # (3) Including gradient checkpointing when the model is loaded in 4/8-bit and some details.
#     new_model = prepare_model_for_kbit_training(
#         model=model,
#         use_gradient_checkpointing=use_gradient_checkpointing)
#     # The above only enables gradient checkpointing for BNB-quantized models
#     # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L81C1-L117C17
#     if use_gradient_checkpointing is True:
#         _enable_gradient_checkpointing(new_model)
#     new_model = get_peft_model(new_model, peft_config)
#     new_model.print_trainable_parameters()
#     if not isinstance(new_model, PeftModelForSequenceClassification):
#         raise TypeError(f"Expected PeftModelForSequenceClassification, but got {type(new_model)}")
#     return new_model

