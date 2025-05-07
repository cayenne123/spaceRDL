import json
import os
# os.environ["USE_TF"] = "0"
# os.environ["TRANSFORMERS_NO_TF"] = "1"
from eval_tool.rough_compute import compute_standard_rough
from eval_tool.bleu_compute import calculate_bleu_scores, calculate_meteor_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import word_tokenize
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# sbert_model = model.to(device)
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def embed_device(name: str) -> np.ndarray:
    """
    使用 Sentence-BERT 将设备名 name 转成向量：
      1) 直接调用 model.encode，返回 (hidden_size,) 的 numpy 向量
    """
    # encode 返回 numpy 数组，默认为 L2 归一化后的向量
    emb = sbert_model.encode(name, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # 已经是 L2 归一化，直接点积就是余弦

"""
对于每一个样本，拆分成<TimeConstraint, Pattern, Device, Variable>, 计算预测标签和真实标签的precision-sample、recall-sample, f1-sampe
对于Device, Variable, 认为数量、语义一致则正确
true: At(150s) SendData SS {J2000Speed}; <At(150s), SendData, SS, J2000Speed>
predict： SendData SS {attirate}; <SendData, SS, J2000Speed>
precision: 3/3
recall: 3/4
"""
def smart_tokenize(text):
    """
    安全地进行 token 切分：
    - 如果 text 是 dict，则尝试提取其中第一个非空字符串值
    - 然后使用正则进行分词
    """
    if isinstance(text, dict):
        for v in text.values():
            if isinstance(v, str) and v.strip():
                text = v
                break
        else:
            return []  # 所有值都不是字符串，返回空 token 列表

    if not isinstance(text, str):
        return []

    return re.findall(r"\w+|\S", text)

def compute_bleu_batch(candidates, references):
    """
    计算每一对 candidate 和 reference 的 BLEU 分数
    :param candidates: list[str]，生成的句子列表
    :param references: list[str]，标准答案列表
    :return: bleu_score 分数列表
    """
    nltk.download('punkt_tab')
    tokenized_candidates = [smart_tokenize(cand) for cand in candidates]
    tokenized_references = [[smart_tokenize(ref)] for ref in references]  # 多一层列表结构
    # print(tokenized_candidates)
    # print(tokenized_references)
    smoothie = SmoothingFunction().method4

    bleu_score = corpus_bleu(
        tokenized_references,
        tokenized_candidates,
        weights=(0.5, 0.5),
        smoothing_function=smoothie
    )

    return bleu_score

def parse_automic_func(text):
    if isinstance(text, dict):
        # 提取第一个值为字符串的字段
        text_values = [v for v in text.values() if isinstance(v, str) and v.strip()]
        if text_values:
            text = text_values[0]
        else:
            # 所有 value 都不是合法字符串，直接返回空结果
            return {
                "TimeConstraint": None,
                "ReqCapByForm": None,
                "TimeConsDef": None,
                "CoreFunc": None,
            }

    text = text.split(';')[0]+";"
    # 定义正则表达式
    time_constraint_pattern = r'(?P<time_constraint>\b(?:At|In|After|Over)\s*\([^\)]*\)|\b(?:In|After)\s*\[\s*[^]]*\s*\])'
    time_cons_def_pattern = r'(?P<time_cons_def>\bFinished\s+Within\s+\S+)'
    # req_cap_by_form_pattern = r'\bReqCapBy(?:Table|Formula|NL|PseudoCode|FlowChart)\s+(?P<req_cap_by_form>\S+)'
    req_cap_by_form_pattern = r'\bReqCapBy(?:Table|Formula|NL|PseudoCode|FlowChart)\s+(?P<req_cap_by_form>\{[^}]+\}|\S+)'

    # 顺序匹配
    tc_match = re.search(time_constraint_pattern, text)
    rc_match = re.search(req_cap_by_form_pattern, text)
    tcd_match = re.search(time_cons_def_pattern, text)

    if rc_match:
        req1 = rc_match.group("req_cap_by_form").replace(";", "")
        if req1.startswith("{") and req1.endswith("}"):
            req = [v.strip() for v in req1[1:-1].split(",")]
        else:
            req = [req1.strip()]
    else:
        req = []

    # 初始化结果
    result = {
        "TimeConstraint": tc_match.group("time_constraint") if tc_match else "",
        "ReqCapByForm": req,
        "TimeConsDef": tcd_match.group("time_cons_def") if tcd_match else "",
        "CoreFunc": "",
    }

    # 删除这三部分后，剩下就是核心指令
    core = text
    for m in [tc_match, rc_match, tcd_match]:
        if m:
            core = core.replace(m.group(0), '')
    # 清理多余空格
    result["CoreFunc"] = re.sub(r'\s{2,}', ' ', core).strip()
    result["CoreFunc"] = result["CoreFunc"].replace(";", "")
    return result


def parse_core_func(core_func_text, time_constaint):
    """
    拆分核心语句，提取模式名称、参数1、参数2
    """
    # 正则提取模式名 + 两个参数
    # pattern = r'^(?P<func>\w+)\s+(?P<arg1>[^{}\s]+)\s+(?P<arg2>\{.*?\}|.+)$'
    # pattern = r'^(?P<func>\w+)\s+(?P<arg1>\{.*?\}|\S+)\s+(?P<arg2>\{.*?\}|\S+)$'
    pattern = r'^(?P<func>\w+)\s+(?P<arg1>\{.*?\}|\S+)\s+(?P<arg2>\{.*?\}|\S+);?$'

    match = re.match(pattern, core_func_text.strip())
    if not match:
        return {
            "PatternName": "",
            "Arg1": "",
            "Arg2": "",
            "TimeConstraint": time_constaint["TimeConstraint"],
            "TimeConsDef": time_constaint["TimeConsDef"],
        }

    def extract_list(arg):
        arg = arg.strip()
        if arg.startswith('{') and arg.endswith('}'):
            items = [item.strip() for item in arg[1:-1].split(',')]
            return [i for i in items if i]  # 去除空项
        else:
            return [arg]

    return {
        "PatternName": match.group("func"),
        "Arg1": extract_list(match.group("arg1")),
        "Arg2": extract_list(match.group("arg2")),
        "TimeConstraint": time_constaint["TimeConstraint"],
        "TimeConsDef": time_constaint["TimeConsDef"],
    }


# 0是完全匹配， 1 是可配置 2 是不匹配
def do_match(output, answer, output_req, answer_req):
    if output['PatternName'] != answer['PatternName']:
        return 2
    elif set(output["Arg1"]) == set(answer["Arg1"]) and set(output["Arg2"]) == set(answer["Arg2"]):
        if set(output_req) == set(answer_req):
            return 0
        else:
            return 1
    else:
        return 2

# def compute_precision(output_rdl, answer_rdl, output_core, answer_core):
#     output_set = [output_core['PatternName'], output_core['Arg1'], output_core['Arg2'], output_rdl['TimeConstraint'] + output_rdl['TimeConsDef']]
#     answer_set = [answer_core['PatternName'], answer_core['Arg1'], answer_core['Arg2'], answer_rdl['TimeConstraint'] + answer_rdl['TimeConsDef']]

#     precision = len(output_set.intersection(answer_set)) / len(answer_set)
#     return precision

OUTPUT_DATA_PATTERNS = {
    'OrbitCtrl', 'ProVld', 'DetAtt',
    'AttCtrl', 'Diagnose', 'SwitchMode','ComputeFunc'
}
def extract_fields(core: dict):
    # 同前：拆分出 common_fields 和 device 名列表
    common, devices = set(), []
    common.add(('Pattern', core['PatternName']))
    if core['PatternName'] in OUTPUT_DATA_PATTERNS:
        common.add(('OutputCount', len(core['Arg1'])))
    else:
        devices = list(core['Arg1'])
    common.add(('InputCount', len(core['Arg2'])))
    tc   = core.get('TimeConstraint','').strip()
    tdef = core.get('TimeConsDef','').strip()
    common.add(('TimeAll', (tc, tdef)))
    # 过滤空项
    filtered = {
        kv for kv in common
        if not (kv[0]=='TimeAll' and kv[1]==('',''))
        and not (kv[0] in ('OutputCount','InputCount') and kv[1]==0)
    }
    return filtered, devices

def compute_sample_metrics(output_core: dict, answer_core: dict):
    out_common, out_devs = extract_fields(output_core)
    ans_common, ans_devs = extract_fields(answer_core)

    # 1) Common TP
    tp_common = len(out_common & ans_common)

    # 2) DeviceName TP via SBERT 相似度匹配
    tp_devs = 0
    used = set()
    ans_embeds = {d: embed_device(d) for d in ans_devs}
    for od in out_devs:
        o_emb = embed_device(od)
        # 找最优匹配
        best_sim, best_ad = 0.0, None
        for ad, a_emb in ans_embeds.items():
            if ad in used: continue
            sim = cosine_sim(o_emb, a_emb)
            if sim > best_sim:
                best_sim, best_ad = sim, ad
        if best_sim >= 0.5:
            tp_devs += 1
            used.add(best_ad)

    # 3) 汇总 TP、DN、AN
    tp      = tp_common + tp_devs
    den_out = len(out_common) + len(out_devs)
    den_ans = len(ans_common) + len(ans_devs)

    # 4) 计算指标
    prec = tp / den_out if den_out>0 else 1.0
    rec  = tp / den_ans if den_ans>0 else 1.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    return prec, rec, f1


if __name__ == "__main__":
    inference_path = "D:/LLM-code/spaceRDL/dataset/results/dataset167new/deepseek-chat/inference.json"

    with open(inference_path, 'r', encoding='utf-8') as f:
        predict_list = json.load(f)

    match_res = [0, 0, 0]
    bleu_scores = []
    bleu_1grams = []
    bleu_2grams = []
    meteor_scores = []
    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []
    belu_scores_batch = []
    true_labels = []
    pred_labels = []

    precision_list = [] 
    recall_list = []
    f1_list = []

    for idx, entry in enumerate(predict_list):
        answer = entry['Correct answer']
        prediction = entry['LLM inference']

        output_rdl = parse_automic_func(prediction)
        answer_rdl = parse_automic_func(answer)
        output_core = parse_core_func(output_rdl['CoreFunc'], output_rdl)
        answer_core = parse_core_func(answer_rdl['CoreFunc'], answer_rdl)

        prec, rec, f1 = compute_sample_metrics(output_core, answer_core)

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        # match_type = do_match(output_core, answer_core, output_rdl['ReqCapByForm'], answer_rdl['ReqCapByForm'])
        # if match_type == 0 and (output_rdl['TimeConstraint'] != answer_rdl['TimeConstraint'] or output_rdl['TimeConsDef'] != answer_rdl['TimeConsDef']):
        #     match_type = 1
        # match_res[match_type] += 1

        r1, r2, rl = compute_standard_rough(answer, prediction)
        bleu, b1, b2 = calculate_bleu_scores(answer, prediction)
        meteor = calculate_meteor_score(answer, prediction)
        bleu_batch = compute_bleu_batch([prediction], [answer])

        bleu_scores.append(bleu)
        bleu_1grams.append(b1)
        bleu_2grams.append(b2)
        meteor_scores.append(meteor)
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
        # print(f"match: {match_type}")
        print(f"precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

    # mlb = MultiLabelBinarizer()
    # mlb.fit(true_labels + pred_labels)

    # true_binary = mlb.transform(true_labels)
    # pred_binary = mlb.transform(pred_labels)

    # precision = precision_score(true_binary, pred_binary, average='samples', zero_division=0)
    # recall = recall_score(true_binary, pred_binary, average='samples', zero_division=0)
    # f1 = f1_score(true_binary, pred_binary, average='samples', zero_division=0)
    total = len(predict_list)
    metric_result = {
        "完全匹配": round(match_res[0] / total, 4),
        "可配置": round(match_res[1] / total, 4),
        "不匹配": round(match_res[2] / total, 4),
        "BLEU": round(sum(bleu_scores) / total, 4),
        "BLEU-1": round(sum(bleu_1grams) / total, 4),
        "BLEU-2": round(sum(bleu_2grams) / total, 4),
        "batch-BLEU": round(sum(belu_scores_batch) / total, 4),
        "METEOR": round(sum(meteor_scores) / total, 4),
        "ROUGE-1 F1": round(sum(rouge_1s) / total, 4),
        "ROUGE-2 F1": round(sum(rouge_2s) / total, 4),
        "ROUGE-L F1": round(sum(rouge_Ls) / total, 4),
        'precision_sample': round(sum(precision_list) / total, 4),
        'recall_sample': round(sum(recall_list) / total, 4),
        'f1_sample': round(sum(f1_list) / total, 4)
    }

    print("\n===== Final Averages =====")
    for key, val in metric_result.items():
        print(f"{key}: {val}")

    metric_output_path = os.path.join(os.path.dirname(inference_path), "metric.json")
    with open(metric_output_path, 'w', encoding='utf-8') as f:
        json.dump(metric_result, f, ensure_ascii=False, indent=2)

    print(f"\n指标结果已保存至：{metric_output_path}")
