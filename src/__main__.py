import logging

from .config.config_model import GeneralConfig
from .retrieval.document_preprocessors.preprocess_config import PreprocessConfig
from .evaluation.evaluator import Evaluator


from .config.from_args import Config


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)

    config = GeneralConfig.from_config()
    run_setup = config.setup_run_config()

    if Config.run.indexing:
        run_setup.indexing_service.embed_documents(
            preprocess_config=config.preprocess_config,
            overwrite_existing_collection=True,
        )

    if Config.run.pipeline:
        while True:
            question = input("Please enter your input. To exit, type 'exit': ")
            if question == "exit":
                break
            print(run_setup.pipeline.invoke(question=question))

    if Config.run.evaluation:
        Evaluator.run_single_evaluation(
            config=config,
            run_setup=run_setup,
        )

    if Config.run.evaluation_multi:
        preprocess_configs = PreprocessConfig.from_config_multi()
        Evaluator.run_multi_evaluation(
            general_config=config,
            preprocess_configs=preprocess_configs,
            retriever_num_documents=[1, 2, 3],
        )


if __name__ == "__main__":
    main()
