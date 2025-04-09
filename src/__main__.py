from datetime import datetime
import logging
import os

from .config.config_model import RAGConfig, RunSetupRAG
from .retrieval.document_preprocessors.preprocess_config import PreprocessConfig
from .evaluation.evaluator import Evaluator


from .config.from_args import Config


def setup_logging() -> None:
    # Create log directory if it doesn't exist
    log_dir = "./data/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to stdout
            logging.FileHandler(log_file),  # Log to file
        ],
    )


def main() -> None:
    logging.getLogger()

    if Config.run.indexing:
        config = RAGConfig.from_config()
        run_setup = RunSetupRAG.run_setup(config)
        run_setup.indexing_service.embed_documents(
            preprocess_config=config.preprocess_config,
            overwrite_existing_collection=True,
        )

    if Config.run.pipeline:
        config = RAGConfig.from_config()
        run_setup = RunSetupRAG.run_setup(config)
        while True:
            question = input("Please enter your input. To exit, type 'exit': ")
            if question == "exit":
                break
            run_setup.pipeline.invoke(question=question)

    if Config.run.evaluation:
        Evaluator.run_single_evaluation()

    if Config.run.evaluation_multi:
        preprocess_configs = PreprocessConfig.from_config_multi()
        Evaluator.run_multi_evaluation(
            preprocess_configs=preprocess_configs,
        )


if __name__ == "__main__":
    setup_logging()
    main()
