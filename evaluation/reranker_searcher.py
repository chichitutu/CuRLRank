import os
import logging
import random
from collections import defaultdict
from typing import Dict, Any, Tuple
from FlagEmbedding.abc.evaluation import EvalReranker
from reranker_prompts import WINDOW_SIZE
random.seed(42)
logger = logging.getLogger(__name__)


class ReasoningEvalReranker(EvalReranker):
    def __init__(
            self,
            reranker,
            rerank_top_k: int = 100,
        ):
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.reranker_sample_k = reranker.sample_k
        self.reranker_base_name = os.path.basename(self.reranker.model_name_or_path)
        self.base_reranker_name = self.reranker_base_name
    
    def __str__(self) -> str:
        return self.base_reranker_name
    
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        search_results: Dict[str, Dict[str, float]],
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Dict[str, float]], float, float]:
        dataset_name = kwargs['dataset_name']
        ########### excluded document ids ###########
        qrels = kwargs.get("reranker_qrels")
        if not qrels:
            logger.warning("No qrels provided, so no documents will be excluded.")
            excluded_ids = {}
        else:
            excluded_ids = {
                qid: [docid for docid, score in docs.items() if not score]
                for qid, docs in qrels.items()
            }
        
        ########### truncate search results to top_k ###########
        # Filter out documents with ids in excluded_ids
        for qid, docs in search_results.items():
            excluded = set(excluded_ids.get(qid, [])) - {"N/A"}

            filtered_docs = {
                doc_id: score
                for doc_id, score in docs.items()
                if doc_id not in excluded
            }

            search_results[qid] = dict(
                sorted(filtered_docs.items(), key=lambda x: x[1], reverse=True)[:self.rerank_top_k]
            )

        ########### generate query-docs pairs ###########
        reranker_data = [
            {
                "q_id": qid,
                "q_content": queries[qid],
                "top_docs": [
                    {
                        "d_id": docid,
                        "d_content": corpus[docid]["text"],
                        "score": score,
                    }
                    for docid, score in docs.items()
                    if not (ignore_identical_ids and qid == docid)
                ],
            }
            for qid, docs in search_results.items()
        ]
        
        ########### reranking ###########
        reranker_data, reranker_cache, all_run_time, all_completion_tokens = self.Iterative_Filtering_Ranking_Strategy(reranker_data, dataset_name)
        
        reranked_results = {
            data["q_id"]: {
                doc["d_id"]: float(doc["score"])
                for doc in data["top_docs"]
            }
            for data in reranker_data
        }
        
        return reranked_results, reranker_cache, all_run_time, all_completion_tokens


    def Iterative_Filtering_Ranking_Strategy(self, reranker_data, dataset_name):

        def iter_windows(docs, window_size):
            for i in range(0, len(docs), window_size):
                yield docs[i:i + window_size]
        
        def build_query_docs_pairs(reranker_data, rank_start, rank_end, step):
            query_docs_pairs = []
            for sample in reranker_data:
                active_docs = sorted(sample["top_docs"], key=lambda d: d["score"], reverse=True)[rank_start:rank_end]
                random.shuffle(active_docs)
                for docs in iter_windows(active_docs, step):
                    query_docs_pairs.append({
                        "qid": sample["q_id"],
                        "query": sample["q_content"],
                        "docid": [d["d_id"] for d in docs],
                        "docs": [d["d_content"] for d in docs],
                        "score": [d["score"] for d in docs],
                        "num_docs": len(docs),
                    })
            return query_docs_pairs

        def compute_cache_score(round_number, reranker_cache):
            prev_round = str(round_number - 1)
            cache = reranker_cache.get(prev_round)
            if not cache:
                return 0.0

            total_score = 0.0
            total_num = 0
            for rerank_list in cache.values():
                for d in rerank_list:
                    scores = d.get("scores", [])
                    total_score += sum(scores)
                    total_num += len(scores)

            return (total_score / total_num) + 15

        def build_current_scores(round_number, query_docs_pairs, final_scores_list, flag_list, cache_score, default_score):
            current_score = defaultdict(dict)
            round_cache = defaultdict(list)
            for i, pair_data in enumerate(query_docs_pairs):
                qid = pair_data["qid"]
                doc_ids = pair_data["docid"]
                scores = final_scores_list[i]
                flag = flag_list[i]
                if not flag:
                    fill_score = default_score if round_number == 1 else cache_score
                    scores = [fill_score] * len(doc_ids)

                for d_id, d_score in zip(doc_ids, scores):
                    current_score[qid][d_id] = d_score

                round_cache[qid].append({"doc_ids": doc_ids, "scores": scores, "flag": flag})

            return current_score, round_cache

        def update_reranker_data(round_number, reranker_data, current_score, cache_score):
            for single in reranker_data:
                scores_map = current_score.get(single["q_id"], {})
                for doc in single["top_docs"]:
                    d_id = doc["d_id"]
                    new_score = scores_map.get(d_id)
                    if new_score is None:
                        continue
                    # smoothing score
                    old_score = doc["score"]
                    doc["score"] = (new_score + (round_number - 1) * old_score) / round_number + 5
        
        reranker_cache = defaultdict(str)
        round_number, min_window, default_score, step = 1, 10, 60.0, WINDOW_SIZE
        rank_start, rank_end = 0, self.rerank_top_k
        all_run_time, all_completion_tokens = 0.0, 0.0

        #################### main loop ####################
        while (rank_end - rank_start) > min_window:
            logger.info(f"round_number: {round_number}, rank_start: {rank_start}, rank_end: {rank_end}")
            print(f"round_number: {round_number}, rank_start: {rank_start}, rank_end: {rank_end}")
            query_docs_pairs = build_query_docs_pairs(reranker_data, rank_start, rank_end, step)
            flag_list, final_scores_list, running_time, completion_tokens = self.reranker.compute_score(query_docs_pairs, dataset_name)
            all_run_time += running_time
            all_completion_tokens += completion_tokens
            
            cache_score = compute_cache_score(round_number, reranker_cache)
            current_score, round_cache = build_current_scores(round_number, query_docs_pairs, final_scores_list, flag_list, cache_score, default_score)
            
            update_reranker_data(round_number, reranker_data, current_score, cache_score)
            reranker_cache[str(round_number)] = round_cache
            
            mean_num_above = (rank_end - rank_start) / 2
            if round_number == 1:
                mean_num_above += 10
            rank_end = min(((int(mean_num_above) + step - 1) // step) * step, rank_end)
            round_number += 1  
        avg_completion_tokens = all_completion_tokens / (round_number - 1)
    
        return reranker_data, reranker_cache, all_run_time, avg_completion_tokens