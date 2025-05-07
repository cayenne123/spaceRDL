import json
import os
from eval_tool.rough_compute import compute_standard_rough
from eval_tool.bleu_compute import calculate_bleu_scores
from llm_inference import parse_automic_func, parse_core_func, do_match, compute_bleu_batch

"""
对于每一个样本，拆分成<TimeConstraint, Pattern, Device, Variable>, 计算预测标签和真实标签的precision-sample、recall-sample, f1-sampe
对于Device, Variable, 认为数量、语义一致则正确
true: At(150s) SendData SS {J2000Speed}; <At(150s), SendData, SS, J2000Speed>
predict： SendData SS {attirate}; <SendData, SS, J2000Speed>
precision: 3/3
recall: 3/4
"""

if __name__ == "__main__":
    inference_path = "D:/LLM/SpaceRDL/RDLAPI/RDLAPI/results/deepseek-chat/inference.json"
    with open(inference_path, 'r', encoding='utf-8') as f:
        predict_list = json.load(f)

    match_res = [0, 0, 0]
    bleu_scores = []
    bleu_1grams = []
    bleu_2grams = []
    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []
    belu_scores_batch = []

    for idx, entry in enumerate(predict_list):
        answer = entry['Correct answer']
        prediction = entry['LLM inference']

        output_rdl = parse_automic_func(prediction)
        answer_rdl = parse_automic_func(answer)
        output_core = parse_core_func(output_rdl['CoreFunc'])
        answer_core = parse_core_func(answer_rdl['CoreFunc'])

        match_type = do_match(output_core, answer_core, output_rdl['ReqCapByForm'], answer_rdl['ReqCapByForm'])
        if match_type == 0 and (output_rdl['TimeConstraint'] != answer_rdl['TimeConstraint'] or output_rdl['TimeConsDef'] != answer_rdl['TimeConsDef']):
            match_type = 1
        match_res[match_type] += 1

        r1, r2, rl = compute_standard_rough(answer, prediction)
        bleu, b1, b2 = calculate_bleu_scores(answer, prediction)
        bleu_batch = compute_bleu_batch([prediction], [answer])

        bleu_scores.append(bleu)
        bleu_1grams.append(b1)
        bleu_2grams.append(b2)
        rouge_1s.append(r1)
        rouge_2s.append(r2)
        rouge_Ls.append(rl)
        belu_scores_batch.append(bleu_batch)

        print(f"\n===== Sample {idx + 1} =====")
        print("LLM predicted answer :")
        print(output_rdl)
        print("\nOriginal answer:")
        print(answer_rdl)
        print("ROUGE 分数:")
        print(f"ROUGE-1 F1: {r1:.4f}, ROUGE-2 F1: {r2:.4f}, ROUGE-L F1: {rl:.4f}")
        print("BLEU 分数:")
        print(f"BLEU: {bleu:.4f}, 1-gram: {b1:.4f}, 2-gram: {b2:.4f}, batch: {bleu_batch:.4f}")
        print(f"match: {match_type}")

    total = len(predict_list)
    metric_result = {
        "完全匹配": round(match_res[0] / total, 4),
        "可配置": round(match_res[1] / total, 4),
        "不匹配": round(match_res[2] / total, 4),
        "BLEU": round(sum(bleu_scores) / total, 4),
        "BLEU-1": round(sum(bleu_1grams) / total, 4),
        "BLEU-2": round(sum(bleu_2grams) / total, 4),
        "batch-BLEU": round(sum(belu_scores_batch) / total, 4),
        "ROUGE-1 F1": round(sum(rouge_1s) / total, 4),
        "ROUGE-2 F1": round(sum(rouge_2s) / total, 4),
        "ROUGE-L F1": round(sum(rouge_Ls) / total, 4),
    }

    print("\n===== Final Averages =====")
    for key, val in metric_result.items():
        print(f"{key}: {val}")

    metric_output_path = os.path.join(os.path.dirname(inference_path), "metric.json")
    with open(metric_output_path, 'w', encoding='utf-8') as f:
        json.dump(metric_result, f, ensure_ascii=False, indent=2)

    print(f"\n指标结果已保存至：{metric_output_path}")
