"""Model loading and augmentation sub-package.

Public API
----------
load_model              : Load a supported LLM and return a ModelWrapperImpl.
ModelWrapperImpl        : Concrete ModelWrapper implementing forward/generate/get_hidden.
TokenizerWrapper        : Consistent tokenizer interface across model families.
apply_infini_attention  : Augment selected attention layers with compressive memory.
reset_infini_memory     : Reset all Infini-attention memory states in a model.
get_infini_trainable_params : Return only Infini-attention adapter parameters.
get_lora_targets        : Recommended LoRA target module names per model family.
MODEL_REGISTRY          : Dict mapping short model names to HuggingFace IDs.
"""

from src.models.infini_attention import (
    apply_infini_attention,
    get_infini_trainable_params,
    reset_infini_memory,
)
from src.models.loader import (
    MODEL_REGISTRY,
    ModelWrapperImpl,
    TokenizerWrapper,
    get_lora_targets,
    load_model,
)

__all__ = [
    "load_model",
    "ModelWrapperImpl",
    "TokenizerWrapper",
    "apply_infini_attention",
    "reset_infini_memory",
    "get_infini_trainable_params",
    "get_lora_targets",
    "MODEL_REGISTRY",
]
