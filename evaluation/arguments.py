from typing import Optional
from dataclasses import dataclass, field
from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs, AbsEvalModelArgs

@dataclass
class CuRLRankEvalArgs(AbsEvalArgs):
    retriever_results_dir: str = field(
        default=None, metadata={"help": "Dir to save or load the retriever results. Default: None"}
    )

    reranker_results_dir: str = field(
        default=None, metadata={"help": "Dir to save or load the reranker results. Default: None"}
    )

@dataclass
class CuRLRankEvalModelArgs(AbsEvalModelArgs):
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl', 'custom']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl", "custom"]}
    )

    reranker_model_class: Optional[str] = field(
        default=None, metadata={"help": "The reranker model class. Available classes: ['encoder-only-base', 'decoder-only-base', 'decoder-only-layerwise', 'decoder-only-lightweight', 'sglang-reasoning']. Default: None. For the custom model, you need to specify the model class.", "choices": ["encoder-only-base", "decoder-only-base", "decoder-only-layerwise", "decoder-only-lightweight", "sglang-reasoning"]}
    )

    reranker_sglang_context_length: int = field(
        default=12288, metadata={"help": "The context length."}
    )
    
    reranker_sglang_tp_size: int = field(
        default=1, metadata={"help": "The tensor parallel size."}
    )

    reranker_sglang_dp_size: int = field(
        default=1, metadata={"help": "The data parallel size."}
    )

    reranker_sglang_seed: int = field(
        default=42, metadata={"help": "The random seed."}
    )

    reranker_max_new_tokens: int = field(
        default=1536, metadata={"help": "The max new tokens."}
    )

    reranker_sample_k: int = field(
        default=1, metadata={"help": "The number of samples to generate for reranking. Default: 1"}
    )

    reranker_enable_full_parallelism: bool = field(
        default=False, metadata={"help": "Whether to enable full parallelism for reranking. Default: False"}
    )

    reranker_enable_thinking: bool = field(
        default=False, metadata={"help": "Whether to enable thinking for reranking. Default: False"}
    )