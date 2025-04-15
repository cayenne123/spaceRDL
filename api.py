import os
import time
from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = "sk-sR8RiK6YYrtk8Rss1b29047069804d108211285c7a25356c"
# os.environ["OPENAI_BASE_URL"] = "https://api.yesapikey.com/v1"
client = OpenAI(
    api_key='sk-1a1093d7857d40dcb93878fe8b21e7bf',
    base_url="https://chat.ecnu.edu.cn/open/api/v1",
)


def read_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None


def ask_gpt(file_path, user_requirement):
    prompt = read_prompt(file_path)
    if not prompt:
        print("文件读取失败，退出。")
        return

    # 替换占位符 {userRequirement} 为用户输入的内容
    prompt = prompt.replace("{userRequirement}", user_requirement)
    gpt_func(prompt)


def gpt_func(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            completion = client.chat.completions.create(
                model="ecnu-reasoner",
                # model="gpt-4o",
                # max_tokens=10000,
                # temperature=1,
                # top_p=1,
                messages=messages
            )
            # 判断有没有补全内容
            print("**************************************************")
            if getattr(completion.choices[0].message, 'content', None):
                content = completion.choices[0].message.content
                print(completion)  # 完整返回值
                print(content)  # 提取补全内容
                break
            else:
                # 如果没有内容，打印错误信息或提示
                # print(completion)
                print('error_wait_2s')
        except:
            pass
        time.sleep(2)


if __name__ == "__main__":
    while True:
        prompt_list = ["定时器初始化", "计算", "控制输出", "数据采集", "数据传输", "自动存储", "自动命令存储"]
        print(prompt_list)
        file_path_num = int(input("请输入所选择的prompt序号，从0开始,输入8结束"))
        if file_path_num == 8:
            break
        file_path = prompt_list[file_path_num] + ".txt"

        user_requirement = input("输入测试的样本")
        if file_path and user_requirement:
            ask_gpt(file_path, user_requirement)
