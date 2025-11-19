"""Base trainer class using Template Method Pattern."""

import os
import traceback
from abc import ABC, abstractmethod
from argparse import Namespace

from yamyam_lab.tools.config import generate_result_path, load_configs
from yamyam_lab.tools.logger import (
    logging_data_statistics,
    logging_experiment_config,
    setup_logger,
)


class BaseTrainer(ABC):
    """Abstract base trainer using Template Method Pattern."""

    def __init__(self, args: Namespace):
        """Initialize trainer with parsed arguments."""
        self.args = args
        self.config = None
        self.preprocess_config = None
        self.result_path = None
        self.data = None
        self.model = None
        self.metric_calculator = None
        self.logger = None

    def train(self) -> None:
        """Template method defining the training workflow."""
        try:
            # Step 1: Load configs
            self.load_configs()

            # Step 2: Setup logger
            self.setup_logger()

            # Step 3: Load data and log statistics
            self.load_data()

            # Step 4: Build model
            self.build_model()

            # Step 5: Build metric calculator
            self.build_metric_calculator()

            # Step 6: Training loop
            self.train_loop()

            # Step 7: Validation evaluation
            self.evaluate_validation()

            # Step 8: Test evaluation
            self.evaluate_test()

            # Step 9: Post-processing
            self.post_process()

        except Exception:
            if self.logger:
                self.logger.error(traceback.format_exc())
            raise

    def load_configs(self) -> None:
        """Load configuration files."""
        config_root = getattr(self.args, "config_root_path", None)
        model = getattr(self.args, "model", "als")

        self.config, self.preprocess_config = load_configs(model, config_root)

        result_root = getattr(self.args, "result_path", None)
        test = getattr(self.args, "test", False)
        self.result_path = generate_result_path(
            model, test, result_root, self.args.postfix
        )

        # Save command to file
        from yamyam_lab.tools.parse_args import save_command_to_file

        save_command_to_file(self.result_path)

    def setup_logger(self) -> None:
        """Setup logger."""
        file_name = self.config.post_training.file_name
        self.logger = setup_logger(os.path.join(self.result_path, file_name.log))

        # Log experiment config after logger is set up
        logging_experiment_config(self.logger, self.args, self.result_path)

    @abstractmethod
    def load_data(self) -> None:
        """Load and prepare dataset. Must be implemented by subclasses.

        After loading data, this method should call log_data_statistics()
        to log the loaded data statistics.
        """
        raise NotImplementedError

    def log_data_statistics(self) -> None:
        """Log data statistics.

        This method is automatically called by load_data() in the template workflow.
        Can be overridden by subclasses if custom logging is needed.
        """
        logging_data_statistics(
            config=self.config,
            data=self.data,
            logger=self.logger,
        )

    @abstractmethod
    def build_model(self) -> None:
        """Build model. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def build_metric_calculator(self) -> None:
        """Build metric calculator. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def train_loop(self) -> None:
        """Training loop. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_validation(self) -> None:
        """Evaluate on validation set. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_test(self) -> None:
        """Evaluate on test set. Must be implemented by subclasses."""
        raise NotImplementedError

    def post_process(self) -> None:
        """Post-processing after training. Can be overridden by subclasses."""
        pass

    def get_top_k_values(self) -> list:
        """Get top-k values for evaluation."""
        top_k_for_pred = self.config.training.evaluation.top_k_values_for_pred
        top_k_for_candidate = self.config.training.evaluation.top_k_values_for_candidate
        return top_k_for_pred + top_k_for_candidate
