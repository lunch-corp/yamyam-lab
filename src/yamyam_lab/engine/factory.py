"""Factory for creating trainers based on model type."""

from argparse import Namespace
from typing import Type

from yamyam_lab.engine.als_trainer import ALSTrainer
from yamyam_lab.engine.base_trainer import BaseTrainer
from yamyam_lab.engine.graph_trainer import GraphTrainer
from yamyam_lab.engine.torch_trainer import TorchTrainer


class TrainerFactory:
    """Factory class for creating appropriate trainers based on model type."""

    # Mapping of model types to trainer classes
    TRAINER_REGISTRY = {
        "als": ALSTrainer,
        "node2vec": GraphTrainer,
        "graphsage": GraphTrainer,
        "metapath2vec": GraphTrainer,
        "lightgcn": GraphTrainer,
        "svd_bias": TorchTrainer,
    }

    @classmethod
    def create_trainer(cls, args: Namespace) -> BaseTrainer:
        """
        Create appropriate trainer based on model type.

        Args:
            args: Parsed arguments containing model type

        Returns:
            Appropriate trainer instance

        Raises:
            ValueError: If model type is not recognized
        """
        model = getattr(args, "model", "als")

        trainer_class = cls.TRAINER_REGISTRY.get(model)

        if trainer_class is None:
            raise ValueError(
                f"Unknown model type: {model}. "
                f"Supported models: {list(cls.TRAINER_REGISTRY.keys())}"
            )

        return trainer_class(args)

    @classmethod
    def register_trainer(
        cls, model_name: str, trainer_class: Type[BaseTrainer]
    ) -> None:
        """
        Register a custom trainer for a model type.

        Args:
            model_name: Name of the model
            trainer_class: Trainer class to register
        """
        if not issubclass(trainer_class, BaseTrainer):
            raise TypeError("trainer_class must be a subclass of BaseTrainer")

        cls.TRAINER_REGISTRY[model_name] = trainer_class
