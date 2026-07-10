import re
import numpy as np
from typing import List

def extract_scores(text: str, expected_num: int):
    blocks = re.findall(
        r"<(?:scores|final_scores)>\s*([\s\S]*?)(?:</(?:scores|final_scores)>|$)",
        text,
        re.I
    )
    # 没闭合标签
    if not blocks:
        blocks = re.findall(
            r"<(?:scores|final_scores)>(.*)",
            text,
            re.S | re.I
        )

    if not blocks:
        return False, []

    candidates = []

    for block in blocks:

        block = block.strip()

        # =========================================================
        # Case 1:
        # [85] [72] [31]
        # =========================================================
        bracket_nums = re.findall(r"\[(\d{1,3})\]", block)

        parsed = None

        if len(bracket_nums) == expected_num:

            parsed = [int(x) for x in bracket_nums]

        # =========================================================
        # Case 2:
        # [1] [85] [2] [72] ...
        # interleaved doc ids
        # =========================================================
        elif len(bracket_nums) == expected_num * 2:

            nums = [int(x) for x in bracket_nums]

            doc_ids = nums[::2]
            scores = nums[1::2]

            expected_doc_ids = list(range(1, expected_num + 1))

            if doc_ids == expected_doc_ids:
                parsed = scores

        # =========================================================
        # Case 3:
        # 85 72 31 10
        # no brackets
        # =========================================================
        else:

            plain_nums = re.findall(r"\b\d{1,3}\b", block)

            if len(plain_nums) == expected_num:
                parsed = [int(x) for x in plain_nums]

        # =========================================================
        # validation
        # =========================================================
        if parsed is None:
            continue

        if not all(0 <= x <= 100 for x in parsed):
            continue

        # =========================================================
        # scoring heuristic
        # =========================================================
        quality = 0

        # 更偏好 bracket format
        quality += len(re.findall(r"\[\d+\]", block))

        # 惩罚 doc 字样
        if "doc" in block.lower():
            quality -= 100

        # 惩罚 prompt copy
        if "score1" in block.lower():
            quality -= 100

        # =========================================================
        # 🚨 新增 heuristic
        # =========================================================

        # 1. 连续递增 [1,2,3,4,5]
        if parsed == list(range(1, len(parsed) + 1)):
            quality -= 1000

        # 2. 小数字(1~9)超过一半
        small_nums = [x for x in parsed if 1 <= x <= 9]

        if len(small_nums) > len(parsed) / 2:
            quality -= 500

        candidates.append((quality, parsed))

    if not candidates:
        return False, []

    best = max(candidates, key=lambda x: x[0])

    return True, best[1]

def compute_final_reward(
    pred_scores: List[float],
    labels: List[int],
    scaling_factor: float = 0.05,
    temperature_coefficient: float = 20.0,
):

    scores = np.asarray(pred_scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)

    pos_scores = scores[labels]
    neg_scores = scores[~labels]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return -1.0

    # -----------------------
    # 1. negative weighting (FIXED) neg_logits取值范围[0,1]左右
    # -----------------------
    neg_logits = scaling_factor * neg_scores

    neg_logits -= np.max(neg_logits)  # stability
    neg_weights = np.exp(neg_logits)
    neg_weights /= (np.sum(neg_weights) + 1e-8)

    # -----------------------
    # 2. pairwise diff (你原来的设计保留)
    # -----------------------
    diff = (pos_scores[:, None] - neg_scores[None, :]) / temperature_coefficient
    pair_prob = 1.0 / (1.0 + np.exp(-diff))

    # -----------------------
    # 3. weighted aggregation
    # -----------------------
    per_pos = (pair_prob * neg_weights[None, :]).sum(axis=1)

    return float(per_pos.mean())

def compute_score(
    predict_str: str, 
    labels: List[int],
    gold_scores: List[float]
):
    format_flag, predict_scores = extract_scores(predict_str, len(labels))
    if format_flag:
        return compute_final_reward(predict_scores, labels)
    else:
        return -1