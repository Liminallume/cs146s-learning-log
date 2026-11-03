import inspect
import json
import os
# 注意：此处适配了使用的 DeepSeek (Anthropic 协议)
import anthropic 
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Tuple

load_dotenv()

# 初始化 LLM 真实对象
claude_client = anthropic.Anthropic(
    api_key="sk-你的真实DeepSeekKey", # 替换你的 Key
    base_url="https://api.deepseek.com/anthropic"
)

# 基础工具：System Prompt
SYSTEM_PROMPT = """
You are a helpful coding assistant whose goal is to help complete coding tasks. 
You have access to a series of tools you can execute. Here are the tools you have access to:

{tool_list_str}

When you want to use a tool reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.
"""

YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"

# 基础工具：相对路径补全函数
def resolve_abs_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path

# ==================== Tools (工具：锤子和扳手) ====================
def read_file_tool(filename: str) -> Dict[str, Any]:             #对读取文件这个工具调用进行真正实现
    """
    Gets the full content of a file provided by the user.     #给LLM写的注释是给LLM写的让他明白这个工具用来干什么，如果LLM要调用这个函数需要传出来什么参数的作用，在程序运行到SystemPrompt的时候反射到这块补全SystemPrompt
    :param filename: The name of the file to read.             
    :return: The full content of the file.
    """
    full_path = resolve_abs_path(filename)                          #调取获得文件路径那个函数为了接下来真正执行读取文件操作
    print(full_path)                                                                #打印出来然用户知道你在读取哪些文件也方便开发者调试
    with open(str(full_path), "r", encoding="utf-8") as f:                                    #真正执行读取文件的操作，读取相对路径里的内容放到f临时文件里面
        content = f.read()                                                        #将内容放在content里面
    return {                                                                             #返回的字符形式:(file_name,file_content)，方便大模型结构化接收上下文
        "file_path": str(full_path),            
        "content": content
    }

def list_files_tool(path: str) -> Dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append({
            "filename": item.name,
            "type": "file" if item.is_file() else "dir"
        })
    return {"path": str(full_path), "files": all_files}

def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file.
    """
    full_path = resolve_abs_path(path)
    p = Path(full_path)
    if old_str == "":
        p.write_text(new_str, encoding="utf-8")
        return {"path": str(full_path), "action": "created_file"}
    original = p.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {"path": str(full_path), "action": "old_str not found"}
    edited = original.replace(old_str, new_str, 1)
    p.write_text(edited, encoding="utf-8")
    return {"path": str(full_path), "action": "edited"}

# TOOL_REGISTRY（工具字典：工具注册表）
TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool
}

# ==================== Context Engineering ====================
def get_tool_str_representation(tool_name: str) -> str:
    # 这是一个“自动生成说明书”的函数。它负责把指定的 Python 函数，扒光成一段纯文本，告诉大模型：这个工具叫什么、干嘛用、需要传什么参数。
    
    tool = TOOL_REGISTRY[tool_name]
    # 从工具注册表（TOOL_REGISTRY）里，通过名字（tool_name）精准拿到对应的“真实函数对象”。接下来的操作全都作用于这个函数对象上。
    
    return f"""
    Name: {tool_name}
    # 告诉大模型工具的名字。
    
    Description: {tool.__doc__}
    # 读取函数自带的三引号注释（__doc__），告诉大模型这个工具能干嘛（比如“读取文件”）。
    
    Signature: {inspect.signature(tool)}
    # 用inspect扫描函数的括号，告诉大模型：“你调用我时，必须按这个格式给我传参数”。
    """

def create_full_system_prompt():                  #这个函数是用来将所有可以调用的tools拼接成字符串，让System prompt来映射的用来让System prompt拼接成给LLM完整的System prompt           [把动态生成的工具说明，塞进静态的剧本模板里。]
    tool_str_repr = ""                                        #这里是定义一个空字符串用来接收接下来遍历所有工具注册表里所有工具的字符串信息（tool is name，desprition，以及调用所需arguments）
    for tool_name in TOOL_REGISTRY:           #遍历所有工具注册表
        tool_str_repr += "TOOL\n === " + get_tool_str_representation(tool_name)  #格式 + 调用工具自动说明书函数：get_tool_str_respretation
        tool_str_repr += f"\n{'='*15}\n"              #格式：将‘=’打印15次 (注：原版双引号在部分环境会报错，这里修复为单引号)
        
    #这块不太理解这个format的意思是可以映射到System prompt那个描述里面的意思吗
    #答案：.format()：Python 的“占位符映射”【字符串格式化】
    #在 Python 里，字符串里的 {} 大括号就是一个“占位符”。平时它就是个空壳，但只要你调用 .format() 方法，它就会开始“填坑”
    return SYSTEM_PROMPT.format(tool_list_str=tool_str_repr)

# ==================== Agent Loop ====================
def execute_llm_call(conversation: List[Dict[str, str]]) -> str:             #执行调用指定模型的操作并且返回拿到LLM的纯文本
    # 提取 system 角色（DeepSeek/Anthropic 协议要求 system 提示词单独传）
    system_content = ""
    messages = []
    for msg in conversation:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            messages.append(msg)

    response = claude_client.messages.create(                                    # claude_client 是你之前建好的“通信客户端（天线）”，在这里发起真实的 API 网络请求。
        model="deepseek-chat",                                                                  # 指明你要调用哪个具体的模型
        system=system_content,
        messages=messages,                                                                       #给LLM输入System Prompt+历史记录+工具结果
        max_tokens=5000,                                                            #限定返回文本长度防止跑偏或烧完token
    )
    return response.content[0].text                                                              # 从返回的复杂 JSON 对象中，精准提取出那一段纯文本回复（纯字符串），交给主循环


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:        #解析LLM返回的东西：（纯文本text，字符串string），以及返回能直接调用tools的：LLM所需的工具表[单个工具[工具名，[LLM按我们规定调用工具返回的各个参数]]]
## 解析LLM返回的纯文本，提取出工具名和参数，返回一个列表，列表里装的是元组(工具名, 参数字典)
    """
    Return list of (tool_name, args) requested in 'tool: name({...})' lines.           
    The parser expects single-line, compact JSON in parentheses.
    """
    invocations = []                                                                                                 #初始化一个空！！列表不是字符串！！用来接收接下来不止一个的工具们（JSON形式返回）
    for raw_line in text.splitlines():                                                                         #按行来对LLM返回的纯文本text做切片处理
        line = raw_line.strip()                                                                                     #取出当前这个切片
        if not line.startswith("tool:"):                                                                         #如果当前这个切片不是以我们要求返回工具所要求的格式：'tool:'了话，就跳过
            continue
        try:
            after = line[len("tool:"):].strip()                                                                  #去掉判断是否还是工具的标准字符串判断，切掉他，获取当前字符串的有用信息
            name, rest = after.split("(", 1)                                                                     #这块不是很理解split的调用
#答案：将'('这个所在切一刀，然后分开tool_name,arguments
            if not rest.endswith(")"):                                                                              #判断LLM是否以我们要求的'()'格式输出的
                continue
            json_str = rest[:-1].strip()                                                                             #取出去掉'('剩下的rest字符串转换为其他工具调用能看懂的JSON格式
#strip()：它不是用来划分 Tuple 的。它的作用是去掉字符串开头和结尾的空格或换行符。大模型输出时往往会带多余的空格，strip() 就是用来打扫卫生的
            args = json.loads(json_str)                                                                           #这个一样加载出当前工具的arguments
            invocations.append((name, args))                                                              #追加到当前的调用字符串里                                     
        except Exception:
            continue
    return invocations                          


def run_coding_agent_loop():                                                       #agent的核心循环，用于每次和LLM交互的每次循环：在每一次循环中将harness分析的命令装作用户发给LLM让LLM工作
    conversation = [{
        "role": "system",
        "content": create_full_system_prompt()
    }]                                                                                                    #先把第一个消息:SystemPrompt发出去
    while True:
        try:
            user_msg = input(f"{YOU_COLOR}You{RESET_COLOR}: ")                 #接受用户的输入信息
        except (EOFError, KeyboardInterrupt):
            break
        conversation.append({                                                             #追加用户输入到conversations
            "role": "user",
            "content": user_msg.strip()
        })
        while True:                                                                                 # 内层循环：专门处理大模型的“多步推理”，直到它不再要求调用工具为止
            assistant_resp = execute_llm_call(conversation)              # 带着完整的上下文（含System Prompt和历史对话）请求大模型
            tool_invocations = extract_tool_invocations(assistant_resp)   #返回从LLM纯文本里提取的所需工具信息（是LLM按我们要求写的（工具名，参数组））
            if not tool_invocations:
                print(f"{ASSISTANT_COLOR}Assistant{RESET_COLOR}: {assistant_resp}")   #如果没有返回参数，说明本轮是chat，直接返回LLM返回文本信息按string输出给用户
                conversation.append({                                                     #将本轮输出追加到上下文历史消息记录
                    "role": "assistant",
                    "content": assistant_resp
                })
                break                                                                                   #返回：针对本次用户的提问与LLM的交互结束
            print(f"{ASSISTANT_COLOR}Assistant{RESET_COLOR}: {assistant_resp}")   #输出LLM要调用的工具信息，因为得让用户实时看到他要进行哪些操作
            conversation.append({                                                           #将操作追加，为了可以以后不重复操作？答案：为了维持大模型的记忆连贯
                "role": "assistant",
                "content": assistant_resp
            })
            for name, args in tool_invocations:                                       #接下来是返回LLM所需的工具进行真正调用的结果，前面判断大模型返回的需要函数调用，然后这块判断需要的是哪个工具
                tool = TOOL_REGISTRY[name]                                  
                resp = []          
                if name == "read_file":                                                       #分别接收工具实现的结果
                    resp = tool(args.get("filename", "."))
                elif name == "list_files":
                    resp = tool(args.get("path", "."))
                elif name == "edit_file":
                    resp = tool(args.get("path", "."), args.get("old_str", ""), args.get("new_str", ""))
                conversation.append({                                                         #这块是重点，把harness也就是前面写的那些程序调用工具执行的结果当作user发给LLM的下次token，让LLM接续工作
                    "role": "user",
                    "content": f"tool_result({json.dumps(resp, ensure_ascii=False)})"               ## 核心伪装术：把本地工具的执行结果，伪装成“用户”的回复喂给大模型，触发它的下一步动作
                })

if __name__ == "__main__":
    run_coding_agent_loop()