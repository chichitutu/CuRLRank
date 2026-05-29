import os
import json
import logging
from typing import Dict, List, Optional, Union

from FlagEmbedding.abc.evaluation import AbsEvaluator
from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr
from FlagEmbedding.abc.evaluation.searcher import EvalRetriever, EvalReranker

from custom_searcher import CustomEvalRetriever

logger = logging.getLogger(__name__)


class CuRLRankEvaluator(AbsEvaluator):
    """
    Evaluator class of Bright 
    """
    @staticmethod
    def compute_metrics(
        qrels: Dict[str, Dict[str, int]],
        search_results: Dict[str, Dict[str, float]],
        k_values: List[int],
    ):
        # remove ex_docids from search_results
        for qid in qrels:
            for docid, relevance in qrels[qid].items():
                if relevance == 0:
                    search_results[qid].pop(docid, None)
        
        ndcg, _map, recall, precision = evaluate_metrics(
            qrels=qrels,
            results=search_results,
            k_values=k_values,
        )
        mrr = evaluate_mrr(
            qrels=qrels,
            results=search_results,
            k_values=k_values,
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores
    
    def evaluate_results(
        self,
        search_results_save_dir: str,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000]
    ):
        eval_results_dict = {}
        for file in os.listdir(search_results_save_dir):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(search_results_save_dir, file)
            data_info, search_results = self.load_search_results(file_path)

            _eval_name = data_info['eval_name']
            assert _eval_name == self.eval_name, f'Mismatch eval_name: {_eval_name} vs {self.eval_name} in {file_path}'

            dataset_name = data_info.get('dataset_name', None)
            qrels = self.data_loader.load_qrels(dataset_name=dataset_name)

            eval_results = self.compute_metrics(
                qrels=qrels,
                search_results=search_results,
                k_values=k_values
            )
            eval_results_dict[dataset_name] = eval_results

        return eval_results_dict
    
    def save_reranker_results(
        self,
        output_path: str,
        eval_name: str,
        dataset_name: str,
        retriever_name: str,
        reranker_name: str,
        search_results: Dict[str, Dict[str, float]],
        running_time: Optional[float] = None,
        completion_tokens: Optional[float] = None,
    ):
        sorted_search_results = {
            outer_k: dict(sorted(inner_dict.items(), key=lambda x: x[1], reverse=True))
            for outer_k, inner_dict in search_results.items()
        }

        data = {
            "eval_name": eval_name,
            "dataset_name": dataset_name,
            "retriever_name": retriever_name,
            "reranker_name": reranker_name,
            "running_time": running_time,
            "completion_tokens": completion_tokens,
            "search_results": sorted_search_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    
    def __call__(
        self,
        retriever_results_dir: str,
        reranker_results_dir: str, 
        retriever: Optional[EvalRetriever] = None,
        reranker: Optional[EvalReranker] = None,
        ignore_identical_ids: bool = False,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        #################### Retrieval Stage ####################
        if isinstance(retriever, CustomEvalRetriever):
            retrieval_results_path = os.path.join(retriever_results_dir, f"{dataset_name}.json")
            _, retriever_results = self.load_search_results(retrieval_results_path)
        else:
            raise NotImplementedError("Only CustomEvalRetriever is supported in the current version.")
        
        corpus = self.data_loader.load_corpus(dataset_name=dataset_name)
        queries = self.data_loader.load_queries(dataset_name=dataset_name)

        #################### Reranking Stage ####################
        os.makedirs(reranker_results_dir, exist_ok=True)
        reranker_results_path = os.path.join(reranker_results_dir, f"{dataset_name}.json")
        cache_results_path = os.path.join(reranker_results_dir, 'cache', f'{dataset_name}.json')
        eval_results_path = os.path.join(reranker_results_dir, 'EVAL', 'eval_results.json')

        if not os.path.exists(reranker_results_path) or self.overwrite:
            kwargs['dataset_name'] = dataset_name
            reranker_results, reranker_cache, running_time, completion_tokens = reranker(
                corpus=corpus,
                queries=queries,
                search_results=retriever_results,
                ignore_identical_ids=ignore_identical_ids,
                **kwargs,
            )

            self.save_reranker_results(
                output_path=reranker_results_path,
                eval_name=self.eval_name,
                dataset_name=dataset_name,
                retriever_name=str(retriever),
                reranker_name=str(reranker),
                search_results=reranker_results,
                running_time=running_time,
                completion_tokens=completion_tokens,
            )
            
            if not os.path.exists(cache_results_path):
                self.output_eval_results_to_json(reranker_cache, cache_results_path)
            if not os.path.exists(eval_results_path) or self.overwrite:
                reranker_eval_results = self.evaluate_results(reranker_results_dir, k_values=k_values)
                self.output_eval_results_to_json(reranker_eval_results, eval_results_path)
        else:
            print(f"{reranker_results_path} already exists and self.overwrite = False, so the reranking process is skipped.")