import logging

from FlagEmbedding.abc.evaluation import AbsEvalRunner

from arguments import CuRLRankEvalArgs, CuRLRankEvalModelArgs
from evaluator import CuRLRankEvaluator
from data_loader import CuRLRankEvalDataLoader

from reranker_searcher import ReasoningEvalReranker

logger = logging.getLogger(__name__)


class CuRLRankEvalRunner(AbsEvalRunner):
    def __init__(self, eval_args: CuRLRankEvalArgs, model_args: CuRLRankEvalModelArgs):
        super().__init__(eval_args, model_args)
        self.eval_args: CuRLRankEvalArgs
        self.model_args: CuRLRankEvalModelArgs
    
    @staticmethod
    def get_models(model_args: CuRLRankEvalModelArgs):
        if model_args.embedder_model_class == "custom":
            from custom_searcher import CustomEmbedder
            embedder = CustomEmbedder(model_name_or_path=model_args.embedder_name_or_path)
        else:
            raise NotImplementedError(f"Embedder model class {model_args.embedder_model_class} not implemented.")

        if model_args.reranker_model_class == "sglang-reasoning":
            from reranker_sglang import SGLangReasoningLLMReranker
            reranker = SGLangReasoningLLMReranker(
                model_name_or_path=model_args.reranker_name_or_path,
                max_new_tokens=model_args.reranker_max_new_tokens,
                batch_size=model_args.reranker_batch_size,
                tp_size=model_args.reranker_sglang_tp_size,
                dp_size=model_args.reranker_sglang_dp_size,
                context_length=model_args.reranker_sglang_context_length,
                random_seed=model_args.reranker_sglang_seed,
                sample_k=model_args.reranker_sample_k,
                enable_thinking=model_args.reranker_enable_thinking,
                enable_full_parallelism=model_args.reranker_enable_full_parallelism,
            )
        else:
            raise NotImplementedError(f"Reranker model class {model_args.reranker_model_class} not implemented.")
        return embedder, reranker
    
    def load_data_loader(self) -> CuRLRankEvalDataLoader:
        data_loader = CuRLRankEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

    def load_evaluator(self) -> CuRLRankEvaluator:
        evaluator = CuRLRankEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )
        return evaluator

    def run(self):
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)
        
        if len(dataset_names) == 0:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the default dataset.")
            self.evaluator(
                retriever_results_dir=self.eval_args.retriever_results_dir,
                reranker_results_dir=self.eval_args.reranker_results_dir,
                retriever=self.retriever,
                reranker=self.reranker,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values
            )
            logger.info(f"{self.eval_args.eval_name} evaluation completed.")
        else:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the following dataset names: {dataset_names}")
            print(f"Running {self.eval_args.eval_name} evaluation on the following dataset names: {dataset_names}")
            for dataset_name in dataset_names:
                evaluator_kwargs = {}
                evaluator_kwargs["reranker_qrels"] = self.data_loader.load_qrels(dataset_name=dataset_name)
                logger.info(f"Running {self.eval_args.eval_name} evaluation on: {dataset_name}")
                print(f"========== Running {self.eval_args.eval_name} evaluation on: {dataset_name} ==========")
                self.evaluator(
                    retriever_results_dir=self.eval_args.retriever_results_dir,
                    reranker_results_dir=self.eval_args.reranker_results_dir,
                    retriever=self.retriever,
                    reranker=self.reranker,
                    ignore_identical_ids=self.eval_args.ignore_identical_ids,
                    k_values=self.eval_args.k_values,
                    dataset_name=dataset_name,
                    **evaluator_kwargs,
                )
                logger.info(f"evaluation on {dataset_name} completed.")
                print(f"evaluation on {dataset_name} completed.")
            print(f"{self.eval_args.eval_name} all datasets evaluate finished")
        # logger.info("Start computing metrics.")
        # self.evaluate_metrics(
        #     search_results_save_dir=self.eval_args.output_dir,
        #     output_method=self.eval_args.eval_output_method,
        #     output_path=self.eval_args.eval_output_path,
        #     metrics=self.eval_args.eval_metrics
        # )

    def load_retriever_and_reranker(self):
        embedder, reranker = self.get_models(self.model_args)
        if self.model_args.embedder_model_class == "custom":
            from custom_searcher import CustomEvalRetriever
            retriever = CustomEvalRetriever(
                embedder,
                search_top_k=self.eval_args.search_top_k,
                overwrite=self.eval_args.overwrite
            )
        else:
            raise NotImplementedError(f"Embedder model class {self.model_args.embedder_model_class} not implemented.")
        
        if reranker is not None and self.model_args.reranker_model_class in ["sglang-reasoning"]:
            reranker = ReasoningEvalReranker(
                reranker,
                rerank_top_k=self.eval_args.rerank_top_k
            )
        else:
            raise NotImplementedError(f"Reranker model class {self.model_args.reranker_model_class} not implemented.")
        
        return retriever, reranker
