import pandas as pd
from openai import OpenAI
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import word_tokenize
import json, ast
import os
import codecs
from lark import Lark, UnexpectedInput

LLM_Model = "deepseek-chat"
shot_num = 1
res_output_dir = "./dataset/results/dataset167new_checker"
client = OpenAI(
    # api_key='sk-1a1093d7857d40dcb93878fe8b21e7bf',
    # base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key = "sk-870beb4daf0542938e2877fd88d77c45",
    base_url = "https://api.deepseek.com", ## deepseek
    # api_key = "sk-6e3903e3e5384aed9fd1ebff218a3c38",
    # base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", ## deepseek
)

def to_snake(name: str) -> str:
    """
    驼峰→蛇形
    """
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_bnf_to_lark(bnf_text: str) -> str:
    """
    将一段 BNF grammar 文本，自动转换为 Lark EBNF grammar 字符串。
    """
    # 1) 找到所有 <NonTerminal>
    nts = set(re.findall(r'<([^>]+)>', bnf_text))
    # 2) 构造映射表
    mapping: Dict[str,str] = { nt: to_snake(nt) for nt in nts }
    
    # 3) 逐一替换 "<X>" → "x_snake"
    g = bnf_text
    for nt, snake in mapping.items():
        g = re.sub(fr'<\s*{re.escape(nt)}\s*>', snake, g)
    
    # 4) ::= → :
    g = re.sub(r'::=', ':', g)
    
    # 5) 'literal' → "literal"
    g = re.sub(r"'([^']*)'", r'"\1"', g)
    
    # 6) 转义分号为终结符
    g = re.sub(r';', r' ";"', g)
    
    # 7) 增加 start 入口，取第一个 rule 的左侧
    first = list(mapping.values())[0]
    grammar_body = f"?start: {first}\n" + g
    
    # 8) 拼接头部
    header = r"""
%import common.WS
%ignore WS
"""
    return header.strip() + "\n\n" + grammar_body


def check_grammar(dsl_text: str, parser: Lark):
    """
    返回 (是否通过, 错误信息)
    """
    try:
        parser.parse(dsl_text)
        return True, ""
    except UnexpectedInput as e:
        return False, f"Syntax error at position {e.pos_in_stream}: {e}"
def check_pattern(expected_pattern: str, core_parse: dict):
    """
    expected_pattern: 来自原始 data['pattern']
    core_parse: parse_core_func 返回的 dict
    """
    actual = core_parse.get("PatternName")
    if actual != expected_pattern:
        return False, f"Expected pattern '{expected_pattern}', but got '{actual}'."
    return True, ""

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


def generate_prompt(requirement, example, bnf_grammar):
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

def _extract_balanced(text: str):
    """从 text 中找第一个平衡的大括号块 {…} 并返回，包括内层嵌套。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def normalize_json_block(raw: str) -> dict:
    """
    1. 去掉 ```json/fences；
    2. 提取第一个平衡的大括号块；
    3. 先尝试 json.loads，若失败再用 ast.literal_eval；
    4. 最终返回 Python dict。
    """
    # 1) 删 fence
    s = re.sub(r"```json\s*|\s*```", "", raw).strip()
    # 2) 栈式提取第一个 {…}
    block = _extract_balanced(s)
    if block is None:
        raise ValueError("No JSON object found")
    # 3) 先 json.loads
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        # 4) Fallback: ast.literal_eval 处理单引号等
        return ast.literal_eval(block)

def get_spaceRDL(text: str) -> str:
    parsed = normalize_json_block(text)

    # 如果 normalize_json_block 已经返回 dict，就直接用它
    if isinstance(parsed, dict):
        data = parsed
    else:
        # 否则它还是一个 str，才用 json.loads
        try:
            data = json.loads(parsed)
        except json.JSONDecodeError:
            return "[ERROR] JSON decode failed"

    if "Output DSL" not in data:
        return "[ERROR] Output DSL not found"
    
    dsl_block = data["Output DSL"].strip()

    if isinstance(dsl_block, dict):
        # 如果包含 "DSL code" 字段，则直接取值
        if "DSL code" in dsl_block:
            dsl_text = dsl_block["DSL code"]
        else:
            # 否则取第一个 key（DeepSeek 常见结构）
            dsl_text = next(iter(dsl_block.keys()))
    else:
        dsl_text = str(dsl_block)

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
    file_path = 'D:/LLM-code/spaceRDL/dataset/dataset167.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # 提取“description”和“pattern” 和"answer"三列
    columns_to_clean = ['description', 'pattern', 'answer']

    # 去除内容中的故意换行
    for col in columns_to_clean:
        df[col] = df[col].map(lambda x: x.replace('\n', ' ').replace('\r', ' ').strip() if isinstance(x, str) else x)

    selected_columns = df[columns_to_clean].to_dict(orient='records')

    device = open_file('D:/LLM-code/spaceRDL/dataset/dict/device.txt')
    dict_data = open_file('D:/LLM-code/spaceRDL/dataset/dict/data.txt')
    bnf_grammar = open_file('D:/LLM-code/spaceRDL/dataset/dict/bnf_grammar.txt')
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

    MAX_RETRIES = 3

    for i, column in enumerate(selected_columns):
        user_requirement = column['description']
        answer_list.append(column['answer'])
        pattern = column['pattern']
        pattern_example = open_file('D:/LLM-code/spaceRDL/dataset/dict/example/' + pattern + '.txt')
        bnf_grammar = open_file('D:/LLM-code/spaceRDL/dataset/dict/grammar/' + pattern + '.txt')
        lark_grammar = open_file('D:/LLM-code/spaceRDL/dataset/dict/grammar_checker/' + pattern + '.txt')
        parser = Lark(
            lark_grammar,
            parser="earley",
            lexer="dynamic_complete",
        )
        prompt = generate_prompt(user_requirement, pattern_example, bnf_grammar)


        for _ in range(MAX_RETRIES):
            res = ask_llm(prompt)
            spaceRDL = get_spaceRDL(res)
            output_list.append(spaceRDL)

            # 1) 语法检查
            ok_syntax, err_syntax = check_grammar(spaceRDL, parser)
            # 2) 提取核心，做模式一致性检查
            core = parse_core_func(parse_automic_func(spaceRDL)['CoreFunc'])
            ok_pattern, err_pattern = check_pattern(pattern, core)
            
            if ok_syntax and ok_pattern:
                break  # 通过检查
            # 否则，拼接错误反馈，生成新 Prompt
            feedback = "The DSL you just output has the following issues:\n"
            if not ok_syntax:
                feedback += f"- Syntax error: {err_syntax}\n"
            if not ok_pattern:
                feedback += f"- Pattern mismatch: {err_pattern}\n"
            feedback += "Please output a corrected SpaceRDL expression that conforms to the provided BNF grammar and matches the required pattern."
            prompt = prompt + "\n\n" + feedback
        else:
            print(f"[Warning] 达到最大重试次数，仍未通过检查，使用最后一次输出。")

        predict_dict = {
            "Requirement": user_requirement,
            "Correct answer": column['answer'],
            "LLM inference": spaceRDL,
            "LLM total answer": res
        }
        predict_list.append(predict_dict)


        # 构建输出路径
        output_dir = os.path.join(res_output_dir, f"{LLM_Model}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"inference.json")

        # 保存结果
        json_data = json.dumps(predict_list, ensure_ascii=False, indent=2)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(json_data)
