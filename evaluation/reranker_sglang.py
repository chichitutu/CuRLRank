import time
import logging
import sglang as sgl
from tqdm import tqdm
from typing import List, Tuple
from utils import get_prompt_template, truncate_texts, extract_scores
from reranker_prompts import RELEVANCE_DEFINITIONS, QUERY_MAX_LEN, DOC_MAX_LEN
logger = logging.getLogger(__name__)


class SGLangReasoningLLMReranker:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int,
        tp_size: int,
        dp_size: int,
        context_length: int,
        random_seed: int,
        batch_size: int,
        sample_k: int,
        enable_thinking: bool,
        enable_full_parallelism: bool,
    ):
        self.reranker_model_class = "sglang-reasoning"
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.context_length = context_length
        self.random_seed = random_seed
        self.max_prompt_length = context_length - max_new_tokens - 64
        self.sample_k = sample_k
        self.enable_thinking = enable_thinking
        self.enable_full_parallelism = enable_full_parallelism
        self.init_flag = False

    def __del__(self):
        if self.init_flag:
            self.model.shutdown()

    def _init_engine(self):
        if not self.init_flag:
            self.model = sgl.Engine(
                model_path=self.model_name_or_path,
                context_length=self.context_length,
                tp_size=self.tp_size,
                dp_size=self.dp_size,
                random_seed=self.random_seed,
            )
            self.tokenizer = self.model.tokenizer_manager.tokenizer
            self.tokenizer.padding_side = "left"
            self.init_flag = True

    def compute_score(
        self,
        query_docs_pairs: List[Tuple[str, str]],
        dataset_name: str,
    ):
        self._init_engine()
        relevance_definition = RELEVANCE_DEFINITIONS[dataset_name]
        query_max_length = QUERY_MAX_LEN
        doc_max_length = DOC_MAX_LEN

        sampling_params = {
            "n": self.sample_k,
            "temperature": 0.25,
            "top_k": 40,
            "repetition_penalty": 1.0,
            "max_new_tokens": self.max_new_tokens,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        if self.enable_full_parallelism:
            sampling_params["n"] = 1

        
        running_time, completion_tokens = 0.0, 0.0
        num_batches = (len(query_docs_pairs) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(query_docs_pairs), self.batch_size), desc="Reasoning Reranking", total=num_batches):
            batch = query_docs_pairs[i:i + self.batch_size]
            batch_queries, batch_docs, batch_docs_num = [], [], []
            for pair in batch:
                batch_queries.append(pair["query"])
                batch_docs.append(pair["docs"])
                batch_docs_num.append(pair["num_docs"])

            batch_queries = truncate_texts(self.tokenizer ,batch_queries, query_max_length, front_ratio=0.7)
            
            flatten_docs = [doc for sublist in batch_docs for doc in sublist]
            truncated_flat_docs = truncate_texts(self.tokenizer, flatten_docs, doc_max_length, front_ratio=0.7)
            
            restored_batch_docs = []
            start_index = 0
            for count in batch_docs_num:
                end_index = start_index + count
                docs_group = truncated_flat_docs[start_index:end_index]
                # format: "[1]: {doc_1} \n[2]: {doc_2} ... \n[10]: {doc_10}"
                docs_text = "\n".join([f"[{i+1}]: {doc}" for i, doc in enumerate(docs_group)])
                restored_batch_docs.append(docs_text)
                start_index = end_index

            batch_docs = restored_batch_docs

            ############ create_prompts ############
            batch_prompts = []
            for idx, query in enumerate(batch_queries):
                prompt = get_prompt_template(relevance_definition, query, batch_docs[idx], batch_docs_num[idx])
                batch_prompts.append(prompt)

            ############ check_prompts_length ############
            truncated_prompts_token_ids = []
            prompts_token_ids = self.tokenizer(batch_prompts)["input_ids"]
            for prompt_token_ids in prompts_token_ids:
                if len(prompt_token_ids) > self.max_prompt_length:
                    print(f"Truncating prompt from {len(prompt_token_ids)} tokens to {self.max_prompt_length} tokens.")
                    prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
                truncated_prompts_token_ids.append(prompt_token_ids)
            truncated_prompts = self.tokenizer.batch_decode(truncated_prompts_token_ids)

            messages = [[{"role": "user", "content": prompt}] for prompt in truncated_prompts]
            input_texts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )

            if self.enable_full_parallelism:
                input_texts = [text for text in input_texts for _ in range(self.sample_k)]

            ############ inference ############
            time_start = time.time()
            outputs = self.model.generate(
                input_texts,
                sampling_params=sampling_params,
            )
            running_time += time.time() - time_start

            ############ extract_scores ############
            reranking_scores = []
            valid_flags = []

            assert len(outputs) == len(batch_prompts) * self.sample_k, (
                f"Expected {len(batch_prompts) * self.sample_k} outputs, "
                f"but got {len(outputs)} outputs."
            )

            for doc_idx, i in enumerate(range(0, len(outputs), self.sample_k)):

                k_outputs = outputs[i:i + self.sample_k]
                docs_num = batch_docs_num[doc_idx]

                sum_scores = [0.0] * docs_num
                valid_count = 0

                for output in k_outputs:
                    completion_tokens += output["meta_info"]["completion_tokens"]

                    flag, score_list = extract_scores(output["text"], docs_num)

                    if not flag:
                        continue

                    valid_count += 1
                    for j, score in enumerate(score_list[:docs_num]):
                        sum_scores[j] += score

                if valid_count:
                    avg_scores = [s / valid_count for s in sum_scores]
                    avg_flag = True
                else:
                    avg_scores = [0.0] * docs_num
                    avg_flag = False

                reranking_scores.append(avg_scores)
                valid_flags.append(avg_flag)

        completion_tokens = completion_tokens / (len(query_docs_pairs) * self.sample_k)
        # List[Bool], List[List[float]], float, float 
        return valid_flags, reranking_scores, running_time, completion_tokens