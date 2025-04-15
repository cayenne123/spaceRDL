import pandas as pd
from openai import OpenAI
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import word_tokenize
from eval_tool.rough_compute import compute_standard_rough
from eval_tool.bleu_compute import calculate_bleu_scores
from eval_tool.codebleu_compute import codebleu_compute
import json
import os

LLM_Model = "deepseek-chat"
shot_num = 1
res_output_dir = "D:/LLM/SpaceRDL/RDLAPI/RDLAPI/results"
client = OpenAI(
    # api_key='sk-1a1093d7857d40dcb93878fe8b21e7bf',
    # base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key="sk-870beb4daf0542938e2877fd88d77c45",
    base_url="https://api.deepseek.com"  ## deepseek
)


def ask_llm(prompt):
    print("开始生成SpaceRDL****************************************")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            completion = client.chat.completions.create(
                model=LLM_Model,
                temperature=0,
                timeout=60,
                messages=messages
            )
            if getattr(completion.choices[0].message, 'content', None):
                content = completion.choices[0].message.content
                # print(completion)
                # print(content)  # 提取补全内容
                res = content
                break
            else:
                # 如果没有内容，打印错误信息或提示
                # print(completion)
                print('error_wait_2s')

        except:
            pass
    return res


def open_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def generate_prompt(requirement, example):
    prompt1 = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/prompt-new.txt')
    prompt1 = prompt1.replace("{device}", device)
    prompt1 = prompt1.replace("{data_dictionary}", dict_data)
    prompt1 = prompt1.replace("{example}", example)
    prompt1 = prompt1.replace("{requirement}", requirement)
    return prompt1


def extract_json(response_content):
    # 匹配 ```json 包裹的代码块
    json_block_pattern = r"```json\s*(\{.*?\}|\[.*?\])\s*```"
    matches = re.findall(json_block_pattern, response_content, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    # 如果没有匹配到代码块，尝试直接匹配裸 JSON
    json_pattern = r"(\{.*?\}|\[.*?\])"
    matches = re.findall(json_pattern, response_content, re.DOTALL)
    if matches:
        for match in matches:
            try:
                # 尝试解析为 JSON
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # 如果还是没有匹配到有效 JSON，返回None
    return None


def get_spaceRDL(text):
    # match = re.search(r'SpaceRDL:\s*(.*?)\s*END', text, re.DOTALL | re.IGNORECASE)
    # if not match:
    #     return "[ERROR] SpaceRDL not found"
    # return match.group(1).strip()
    data = extract_json(text)

    # 提取 "requirement" 字段
    dsl = data["Output DSL"]
    return dsl


def parse_automic_func(text):
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
        "TimeConstraint": tc_match.group("time_constraint") if tc_match else None,
        "ReqCapByForm": req,
        "TimeConsDef": tcd_match.group("time_cons_def") if tcd_match else None,
        "CoreFunc": None,
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


def parse_core_func(core_func_text):
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
            "PatternName": None,
            "Arg1": [],
            "Arg2": []
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
        "Arg2": extract_list(match.group("arg2"))
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


def smart_tokenize(text: str):
    # 用正则切分：字母/数字 和 标点分开
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


if __name__ == "__main__":
    file_path = 'D:/LLM/SpaceRDL/RDLAPI/RDLAPI/test.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 提取“description”和“pattern” 和"answer"三列
    columns_to_clean = ['description', 'pattern', 'answer']

    # 去除内容中的故意换行
    for col in columns_to_clean:
        df[col] = df[col].map(lambda x: x.replace('\n', ' ').replace('\r', ' ').strip() if isinstance(x, str) else x)

    selected_columns = df[columns_to_clean].to_dict(orient='records')

    device = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/dict/device.txt')
    dict_data = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/dict/data.txt')
    output_list = []
    answer_list = []

    match_res = [0, 0, 0]
    predict_list = []

    # 生成SpaceRDL
    bleu_scores = []
    bleu_1grams = []
    bleu_2grams = []
    rouge_1s = []
    rouge_2s = []
    rouge_Ls = []
    codebleu_scores = []
    belu_scores_batch = []

    for i, column in enumerate(selected_columns):
        user_requirement = column['description']
        answer_list.append(column['answer'])
        pattern = column['pattern']
        pattern_example = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/dict/' + pattern + '.txt')
        prompt = generate_prompt(user_requirement, pattern_example)
        res = ask_llm(prompt)
        spaceRDL = get_spaceRDL(res)
        output_list.append(spaceRDL)

        output_rdl = parse_automic_func(spaceRDL)
        answer_rdl = parse_automic_func(column['answer'])
        output_rdl_core = parse_core_func(output_rdl['CoreFunc'])
        answer_rdl_core = parse_core_func(answer_rdl['CoreFunc'])

        predict_dict = {
            "Requirement": user_requirement,
            "Correct answer": column['answer'],
            "LLM inference": spaceRDL,
            "LLM total answer": res
        }
        predict_list.append(predict_dict)

        # 匹配类别
        match_type = do_match(output_rdl_core, answer_rdl_core, output_rdl['ReqCapByForm'], answer_rdl['ReqCapByForm'])
        if match_type == 0:
            if output_rdl['TimeConstraint'] != answer_rdl['TimeConstraint'] or output_rdl['TimeConsDef'] != answer_rdl[
                'TimeConsDef']:
                match_type = 1
        match_res[match_type] += 1

        # 评估指标
        rouge1_f1, rouge2_f1, rougeL_f1 = compute_standard_rough(column['answer'], spaceRDL)
        bleu_score, bleu_1gram, bleu_2gram = calculate_bleu_scores(column['answer'], spaceRDL)
        # codebleu_res = codebleu_compute(column['answer'], spaceRDL)
        bleu_score_batch = compute_bleu_batch([spaceRDL], [column['answer']])

        # 指标累计
        bleu_scores.append(bleu_score)
        bleu_1grams.append(bleu_1gram)
        bleu_2grams.append(bleu_2gram)
        belu_scores_batch.append(bleu_score_batch)
        rouge_1s.append(rouge1_f1)
        rouge_2s.append(rouge2_f1)
        rouge_Ls.append(rougeL_f1)
        # codebleu_scores.append(codebleu_res["codebleu"])

        # 打印中间结果
        print(f"\nLLM predicted answer :\n{output_rdl}")
        print(f"\nOriginal answer:\n{answer_rdl}")
        print("ROUGE 分数:")
        print(f"ROUGE-1 F1: {rouge1_f1:.4f}, ROUGE-2 F1: {rouge2_f1:.4f}, ROUGE-L F1: {rougeL_f1:.4f}")
        print("BLEU 分数:")
        print(
            f"BLEU: {bleu_score:.4f}, 1-gram: {bleu_1gram:.4f}, 2-gram: {bleu_2gram:.4f}, batch: {bleu_score_batch:.4f}")
        print(f"match: {match_type}")

        # 构建输出路径
        output_dir = os.path.join(res_output_dir, f"{LLM_Model}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"inference.json")

        # 保存结果
        json_data = json.dumps(predict_list, ensure_ascii=False, indent=2)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(json_data)

    # ==== 平均指标输出 ====
    print("\n=== Final Statistics ===")
    print("完全匹配：", match_res[0] / len(output_list))
    print("可配置：", match_res[1] / len(output_list))
    print("不匹配：", match_res[2] / len(output_list))
    print(f"BLEU 平均：{sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"1-gram BLEU 平均：{sum(bleu_1grams) / len(bleu_1grams):.4f}")
    print(f"2-gram BLEU 平均：{sum(bleu_2grams) / len(bleu_2grams):.4f}")
    print(f"batch BLEU 平均：{sum(belu_scores_batch) / len(belu_scores_batch):.4f}")
    print(f"ROUGE-1 F1 平均：{sum(rouge_1s) / len(rouge_1s):.4f}")
    print(f"ROUGE-2 F1 平均：{sum(rouge_2s) / len(rouge_2s):.4f}")
    print(f"ROUGE-L F1 平均：{sum(rouge_Ls) / len(rouge_Ls):.4f}")
    # print(f"CodeBLEU 平均：{sum(codebleu_scores)/len(codebleu_scores):.4f}")

    # 构建指标汇总字典
    metric_result = {
        "完全匹配": round(match_res[0] / len(output_list), 4),
        "可配置": round(match_res[1] / len(output_list), 4),
        "不匹配": round(match_res[2] / len(output_list), 4),
        "BLEU": round(sum(bleu_scores) / len(bleu_scores), 4),
        "BLEU-1": round(sum(bleu_1grams) / len(bleu_1grams), 4),
        "BLEU-2": round(sum(bleu_2grams) / len(bleu_2grams), 4),
        "batch-BLEU": round(sum(belu_scores_batch) / len(belu_scores_batch), 4),
        "ROUGE-1 F1": round(sum(rouge_1s) / len(rouge_1s), 4),
        "ROUGE-2 F1": round(sum(rouge_2s) / len(rouge_2s), 4),
        "ROUGE-L F1": round(sum(rouge_Ls) / len(rouge_Ls), 4),
        # "CodeBLEU": round(sum(codebleu_scores) / len(codebleu_scores), 4)
    }

    # 输出路径
    metric_output_path = os.path.join(output_dir, "metric.json")

    # 保存为 JSON 文件
    with open(metric_output_path, "w", encoding="utf-8") as f:
        json.dump(metric_result, f, ensure_ascii=False, indent=2)

    print(f"\n指标结果已保存至：{metric_output_path}")
