import os
import time
import json
import pandas as pd
from openai import OpenAI

LLM_Model = "deepseek-chat"
res_output_dir = "D:/LLM/SpaceRDL/RDLAPI/RDLAPI/results"
client = OpenAI(
    # api_key='sk-1a1093d7857d40dcb93878fe8b21e7bf',
    # base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key = "sk-870beb4daf0542938e2877fd88d77c45",
    base_url = "https://api.deepseek.com" ## deepseek
)

def ask_llm(prompt):
    print("开始生成****************************************")
    messages = [
        {"role": "system", "content": "你是一个专业的航天嵌入式软件需求分析师。"},
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

# 输入/输出路径
excel_path = 'D:/LLM/SpaceRDL/RDLAPI/RDLAPI/test.xlsx'
# df = pd.read_excel(file_path, sheet_name='Sheet1')

# # 提取“description”和“pattern” 和"answer"三列
# columns_to_clean = ['description', 'pattern', 'answer']

res_output_dir = "D:/LLM/SpaceRDL/RDLAPI/RDLAPI/results"
output_dir = os.path.join(res_output_dir, f"{LLM_Model}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
json_output_path = os.path.join(output_dir, "reasoning_output.json")

# DSL 定义：BNF、设备库、数据字典
bnf = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/dict/bnf.txt')
device = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/dict/device.txt')
dict_data = open_file('D:/LLM/SpaceRDL/RDLAPI/RDLAPI/dict/data.txt')

# 模板
template = """假设你是一个航天嵌入式软件需求分析师，下面是一段需求和对应的DSL表达：
<需求>
{requirement}
</需求>
<DSL表达>
{dsl_expr}
</DSL表达>

DSL 的 BNF 定义：
{bnf}

设备库：
{device_library}

数据字典：
{data_dictionary}

请根据上述信息，反向推理从需求到DSL表达的思考过程。分析应模拟人类思考，每一步按序号(1, 2, 3, ...)展开，并包含：
1. 解构系统输入输出模型及接口协议
2. 解析核心数据结构
3. 分析具体航天场景需求

注意：
- 推理过程中不能包含 DSL 或代码片段
- 最终形成的推理过程放到<推理过程>标签中
"""

# 读取 Excel
df = pd.read_excel(excel_path)

results = []
for idx, row in df.iterrows():
    requirement = row["description"]
    dsl_expr = row["answer"]
    prompt = template.format(
        requirement=requirement,
        dsl_expr=dsl_expr,
        bnf=bnf.strip(),
        device_library=device.strip(),
        data_dictionary=dict_data.strip()
    )
    
    # 调用大模型
    reasoning = ask_llm(prompt)
    
    results.append({
        "requirement": requirement,
        "dsl": dsl_expr,
        "reasoning": reasoning
    })
    # 避免触发速率限制
    time.sleep(1)

    # 写入 JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print(f"推理结果已保存到：{json_output_path}")
