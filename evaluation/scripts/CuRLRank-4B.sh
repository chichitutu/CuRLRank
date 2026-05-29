ROOT_DIR="/mnt/work/yunxiao_4/CuRLRank/evaluation"
# nohup bash scripts/CuRLRank-4B.sh > logs/run.log 2>&1 &

reranker_models=("CuRLRank-4B")

for reranker_model in "${reranker_models[@]}"; do

    echo "================================================="
    echo "Running reranker: ${reranker_model}"
    echo "================================================="

    eval_name="bright"
    retriever_model="reasonir-8b"
    retriever_data="bright_reasonir-8b_top100"

    echo "   ================================================="
    echo "   Running benchmark: ${eval_name}"
    echo "   ================================================="

    dataset_names=('biology' 'earth_science' 'economics' 'robotics' 'psychology' 
                     'stackoverflow' 'sustainable_living' 'leetcode' 'pony' 'aops' 
                     'theoremqa_questions' 'theoremqa_theorems')

    dataset_dir="${ROOT_DIR}/data/text/${eval_name}"
    embedder_path="${ROOT_DIR}/model/${retriever_model}"
    reranker_path="${ROOT_DIR}/model/${reranker_model}"
    retriever_results_dir="${ROOT_DIR}/data/retrieval/${retriever_data}"
    reranker_results_dir="${ROOT_DIR}/data/reranking/${retriever_data}/${reranker_model}"
    log_dir="${ROOT_DIR}/logs/${retriever_data}/${reranker_model}"
    mkdir -p "${reranker_results_dir}"
    mkdir -p "${log_dir}"

    python main.py \
        --eval_name "${eval_name}" \
        --dataset_dir "${dataset_dir}" \
        --retriever_results_dir "${retriever_results_dir}" \
        --reranker_results_dir "${reranker_results_dir}" \
        --dataset_names "${dataset_names[@]}" \
        --search_top_k 2000 \
        --overwrite True \
        --k_values 1 10 100 \
        --embedder_name_or_path "${embedder_path}" \
        --embedder_model_class custom \
        --reranker_name_or_path "${reranker_path}" \
        --reranker_model_class sglang-reasoning \
        --rerank_top_k 100 \
        --reranker_sample_k 1 \
        --reranker_batch_size 2048 \
        --reranker_sglang_tp_size 1 \
        --reranker_sglang_dp_size 4 \
        --reranker_sglang_seed 42 \
        --reranker_sglang_context_length 12288 \
        --reranker_max_new_tokens 1536 \
        --reranker_enable_thinking False \
        --reranker_enable_full_parallelism False \
        > "${log_dir}/${eval_name}.log" 2>&1


    eval_name="r2med"
    retriever_model="e5-mistral-7b-instruct"
    retriever_data="r2med_e5-mistral-7b-instruct_top100"

    echo "   ================================================="
    echo "   Running benchmark: ${eval_name}"
    echo "   ================================================="

    dataset_names=('r2med_Biology' 'r2med_Bioinformatics' 'r2med_Medical-Sciences' 'r2med_MedXpertQA-Exam'
                    'r2med_MedQA-Diag' 'r2med_PMC-Treatment' 'r2med_PMC-Clinical' 'r2med_IIYi-Clinical')

    dataset_dir="${ROOT_DIR}/data/text/${eval_name}"
    embedder_path="${ROOT_DIR}/model/${retriever_model}"
    reranker_path="${ROOT_DIR}/model/${reranker_model}"
    retriever_results_dir="${ROOT_DIR}/data/retrieval/${retriever_data}"
    reranker_results_dir="${ROOT_DIR}/data/reranking/${retriever_data}/${reranker_model}"
    log_dir="${ROOT_DIR}/logs/${retriever_data}/${reranker_model}"
    mkdir -p "${reranker_results_dir}"
    mkdir -p "${log_dir}"

    python main.py \
        --eval_name "${eval_name}" \
        --dataset_dir "${dataset_dir}" \
        --retriever_results_dir "${retriever_results_dir}" \
        --reranker_results_dir "${reranker_results_dir}" \
        --dataset_names "${dataset_names[@]}" \
        --search_top_k 2000 \
        --overwrite True \
        --k_values 1 10 100 \
        --embedder_name_or_path "${embedder_path}" \
        --embedder_model_class custom \
        --reranker_name_or_path "${reranker_path}" \
        --reranker_model_class sglang-reasoning \
        --rerank_top_k 100 \
        --reranker_sample_k 1 \
        --reranker_batch_size 2048 \
        --reranker_sglang_tp_size 1 \
        --reranker_sglang_dp_size 4 \
        --reranker_sglang_seed 42 \
        --reranker_sglang_context_length 12288 \
        --reranker_max_new_tokens 1536 \
        --reranker_enable_thinking False \
        --reranker_enable_full_parallelism False \
        > "${log_dir}/${eval_name}.log" 2>&1

    eval_name="beir"
    retriever_model="bm25"
    retriever_data="beir_bm25_top100"

    echo "   ================================================="
    echo "   Running benchmark: ${eval_name}"
    echo "   ================================================="

    dataset_names=('covid' 'dbpedia' 'scifact' 'nfcorpus' 'signal' 'robust04' 'news')

    dataset_dir="${ROOT_DIR}/data/text/${eval_name}"
    embedder_path="${ROOT_DIR}/model/${retriever_model}"
    reranker_path="${ROOT_DIR}/model/${reranker_model}"
    retriever_results_dir="${ROOT_DIR}/data/retrieval/${retriever_data}"
    reranker_results_dir="${ROOT_DIR}/data/reranking/${retriever_data}/${reranker_model}"
    log_dir="${ROOT_DIR}/logs/${retriever_data}/${reranker_model}"
    mkdir -p "${reranker_results_dir}"
    mkdir -p "${log_dir}"

    python main.py \
        --eval_name "${eval_name}" \
        --dataset_dir "${dataset_dir}" \
        --retriever_results_dir "${retriever_results_dir}" \
        --reranker_results_dir "${reranker_results_dir}" \
        --dataset_names "${dataset_names[@]}" \
        --search_top_k 2000 \
        --overwrite True \
        --k_values 1 10 100 \
        --embedder_name_or_path "${embedder_path}" \
        --embedder_model_class custom \
        --reranker_name_or_path "${reranker_path}" \
        --reranker_model_class sglang-reasoning \
        --rerank_top_k 100 \
        --reranker_sample_k 1 \
        --reranker_batch_size 2048 \
        --reranker_sglang_tp_size 1 \
        --reranker_sglang_dp_size 4 \
        --reranker_sglang_seed 42 \
        --reranker_sglang_context_length 12288 \
        --reranker_max_new_tokens 1536 \
        --reranker_enable_thinking False \
        --reranker_enable_full_parallelism False \
        > "${log_dir}/${eval_name}.log" 2>&1
done