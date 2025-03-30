export CUDA_VISIBLE_DEVICES=1


# /shared/eng/pj20/lmr_model/ctgov_3b_transfer_1/actor/global_step_100
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/SearchEngine/eval_ctgov.py \
    --model_path /shared/eng/pj20/lmr_model/ctgov_32_transfer/actor/global_step_600 \
    --data_path data/search_engine/ctgov_32/test_full.parquet \
    --model_name ctgov_32_transfer_600