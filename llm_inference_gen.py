import pandas as pd
from openai import OpenAI
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import word_tokenize
from eval_tool.rough_compute import compute_standard_rough
from eval_tool.bleu_compute import calculate_bleu_scores
from eval_tool.codebleu_compute import codebleu_compute
import json, re, textwrap
import os
import codecs

LLM_Model = "deepseek-chat"
shot_num = 1
res_output_dir = "./dataset/results"
client = OpenAI(
    # api_key='sk-1a1093d7857d40dcb93878fe8b21e7bf',
    # base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key = "sk-870beb4daf0542938e2877fd88d77c45",
    base_url = "https://api.deepseek.com" ## deepseek
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
    prompt1 = open_file('D:/LLM-code/spaceRDL/prompt-new.txt')
    prompt1 = prompt1.replace("{device}", device)
    prompt1 = prompt1.replace("{data_dictionary}", dict_data)
    prompt1 = prompt1.replace("{example}", example)
    prompt1 = prompt1.replace("{requirement}", requirement)
    prompt1 = prompt1.replace("{bnf_grammar}", bnf_grammar)
    return prompt1


def fix_encoding(obj):
    if isinstance(obj, dict):
        return {fix_encoding(k): fix_encoding(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_encoding(i) for i in obj]
    elif isinstance(obj, str):
        try:
            return obj.encode('latin1').decode('utf-8')
        except:
            return obj
    return obj

# def extract_json(response_content):
#     # 匹配 ```json 包裹的代码块
#     json_block_pattern = r"```json\s*(\{.*?\}|\[.*?\])\s*```"
#     matches = re.findall(json_block_pattern, response_content, re.DOTALL)
#     if matches:
#         for match in matches:
#             try:
#                 return json.loads(match)
#             except json.JSONDecodeError:
#                 continue
#     # 如果没有匹配到代码块，尝试直接匹配裸 JSON
#     json_pattern = r"(\{.*?\}|\[.*?\])"
#     matches = re.findall(json_pattern, response_content, re.DOTALL)
#     if matches:
#         for match in matches:
#             try:
#                 # 尝试解析为 JSON
#                 return json.loads(match)
#             except json.JSONDecodeError:
#                 continue

#     # 如果还是没有匹配到有效 JSON，返回None
#     return None

try:
    import json5, yaml
except ImportError:
    json5 = yaml = None

def _sanitize(raw: str) -> str:
    """尽量不破坏内容地做最小清洗"""
    s = raw.strip()

    # 1) 去掉行尾注释  // ... 或  # ...
    s = re.sub(r"[ \t]*//.*?$", "", s, flags=re.M)
    s = re.sub(r"[ \t]*#.*?$",  "", s, flags=re.M)

    # 2) 去掉尾逗号   {"a":1,} -> {"a":1}
    s = re.sub(r",(\s*[}\]])", r"\1", s)

    # 3) 把非法 \' 替成普通单引号
    s = s.replace("\\'", "'")

    return s

def extract_json(content: str):
    # ① 先找 ```json ... ``` 代码块
    blocks = re.findall(r"```json(.*?)```", content, flags=re.S | re.I)
    # 如果没包 json 标签，退而求其次找 ``` ... ```
    if not blocks:
        blocks = re.findall(r"```(.*?)```", content, flags=re.S)
    # 如果还没找到，就把全文当候选
    if not blocks:
        blocks = [content]

    # 对每个候选块：
    for blk in blocks:
        # ② 截取到首个完整 {...} 或 [...]
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", blk)
        if not m:
            continue
        candidate = _sanitize(m.group(1))

        # ③ 多轮解析尝试
        for loader in (
            lambda x: json.loads(x),           # 严格 JSON
            (lambda x: json5.loads(x)) if json5 else None,  # JSON5
            (lambda x: yaml.safe_load(x)) if yaml else None # YAML
        ):
            if loader is None:                # 对应库未安装
                continue
            try:
                return loader(candidate)
            except Exception:
                pass  # 换下一种 loader

        # 打印 / 记录错误信息帮助调试
        try:
            json.loads(candidate)
        except json.JSONDecodeError as e:
            print("JSON 解析失败，起始位置行列:", e.lineno, e.colno, "-", e.msg)

    return None


def get_spaceRDL(text: str) -> str:
    data = extract_json(text)
    if not data or "Output DSL" not in data:
        return "[ERROR] Output DSL not found"

    dsl_block = data["Output DSL"]

    if isinstance(dsl_block, dict):
        # 如果包含专门的 "DSL code" 字段，优先取该值
        if "DSL code" in dsl_block:
            dsl_text = dsl_block["DSL code"]
        else:
            # DeepSeek 可能直接用 key 作代码
            dsl_text = next(iter(dsl_block.keys()))
   # 或 "\n".join(...) 按行拼接
    else:
        # 其余情况直接转成字符串
        dsl_text = str(dsl_block)
    if isinstance(dsl_text, list):
        # 如果是列表，把所有元素转成字符串后拼接
        dsl_text = " ".join(map(str, dsl_text))   

    return dsl_text.strip()




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


if __name__ == "__main__":
    # Load Excel
    file_path = './dataset/dataset1338.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    columns_to_clean = ['description', 'pattern']
    for col in columns_to_clean:
        df[col] = df[col].map(lambda x: x.replace('\n', ' ').replace('\r', ' ').strip() if isinstance(x, str) else x)
    selected_columns = df[columns_to_clean].to_dict(orient='records')

    # Load dictionaries
    device = open_file('./dataset/dict/device.txt')
    dict_data = open_file('./dataset/dict/data.txt')
    bnf_grammar = open_file('./dataset/dict/bnf_grammar.txt')

    # Prepare output directory and resume state
    output_dir = os.path.join(res_output_dir, LLM_Model, file_path.split('/')[-1].split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    inference_path = os.path.join(output_dir, "inference.json")
    if os.path.exists(inference_path):
        with open(inference_path, 'r', encoding='utf-8') as f:
            predict_list = json.load(f)
        start_idx = len(predict_list)
        print(f"Resuming from index {start_idx}...")
    else:
        predict_list = []
        start_idx = 0

    # Main loop: resume from start_idx
    for idx in range(start_idx, len(selected_columns)):
        column = selected_columns[idx]
        user_requirement = column['description']
        # answer_list = column['answer']
        pattern = column['pattern']
        example_path = os.path.join('./dataset/dict/example', f"{pattern}.txt")
        with open(example_path, 'r', encoding='utf-8') as ef:
            pattern_example = ef.read()
        prompt = generate_prompt(user_requirement, pattern_example)
        res = ask_llm(prompt)
        spaceRDL = get_spaceRDL(res)

        # Build record
        record = {
            "Requirement": user_requirement,
            # "Correct answer": answer_list,
            "LLM inference": spaceRDL,
            "LLM total answer": res
        }
        predict_list.append(record)

        # Save after each request to allow safe resume
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(predict_list, f, ensure_ascii=False, indent=2)
        print(f"Saved inference for index {idx}")

    print("All done. Total records:", len(predict_list))