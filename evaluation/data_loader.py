import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict
from FlagEmbedding.abc.evaluation import AbsEvalDataLoader
logger = logging.getLogger(__name__)


class CuRLRankEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        if self.eval_name == "bright":
            dataset_names = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living",
                            "leetcode", "pony", "aops", "theoremqa_questions", "theoremqa_theorems"]
        elif self.eval_name == "r2med":
            dataset_names = ["r2med_Biology", "r2med_Bioinformatics", "r2med_Medical-Sciences", 
                            "r2med_MedXpertQA-Exam", "r2med_MedQA-Diag", "r2med_PMC-Treatment",
                            "r2med_PMC-Clinical", "r2med_IIYi-Clinical"]
        elif self.eval_name == "beir":
            dataset_names = ["covid", "dbpedia", "scifact",
                            "nfcorpus", "signal", "robust04", "news"]
        else:
            raise ValueError(f"Invalid task type: {self.eval_args.task_type}")
        return dataset_names
    
    def _load_local_corpus(self, save_dir: str, dataset_name: str) -> datasets.DatasetDict:
        corpus_path = os.path.join(save_dir, "corpus.jsonl")
        corpus_dict = defaultdict(str)
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {self.eval_name} {dataset_name} corpus from local"):
                data = json.loads(line.strip())
                docid, text = data["id"], data["text"]
                corpus_dict[docid] = {"text": text}
        return datasets.DatasetDict(corpus_dict)

    def _load_local_qrels(self, save_dir: str, dataset_name: str, split: Optional[str] = None) -> datasets.DatasetDict:
        qrels_path = os.path.join(save_dir, "qrels.jsonl")
        qrels_dict = defaultdict(dict)
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {self.eval_name} {dataset_name} qrels from local"):
                data = json.loads(line.strip())
                qid, docid, relevance = data["qid"], data["docid"], data["relevance"]
                qrels_dict[qid][docid] = relevance
        return datasets.DatasetDict(qrels_dict)
    
    def _load_local_queries(self, save_dir: str, dataset_name: str, split: Optional[str] = None) -> datasets.DatasetDict:
        queries_path = os.path.join(save_dir, "queries.jsonl")
        queries_dict = defaultdict(str)
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {self.eval_name} {dataset_name} queries from local"):
                data = json.loads(line.strip())
                qid, query = data["id"], data["text"]
                queries_dict[qid] = query
        return datasets.DatasetDict(queries_dict)

    def available_splits(self, dataset_name: str) -> List[str]:
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ]