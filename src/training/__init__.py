"""Training sub-package.

Public API
----------
LoRATrainer         : Main training loop for frozen LLM + LoRA + reservoir interface.
LoRATrainingConfig  : Full training configuration dataclass.
CurriculumScheduler : Progressive task-difficulty scheduler for curriculum training.
CurriculumConfig    : Configuration for the multi-stage curriculum.
CurriculumDataPipeline : DataPipeline that mixes data according to curriculum stage.
"""

from src.training.curriculum import CurriculumConfig, CurriculumDataPipeline
from src.training.lora_trainer import LoRATrainer, LoRATrainingConfig

__all__ = [
    "LoRATrainer",
    "LoRATrainingConfig",
    "CurriculumConfig",
    "CurriculumDataPipeline",
]
