from langchain.schema import HumanMessage
from utils import parse_interaction
from state import BrickState
from builder import build_graph_with_interaction
from save_dir_name import get_save_dir
from dotenv import load_dotenv
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import json
import hashlib
from collections import OrderedDict

load_dotenv(dotenv_path='brick_test_config.env')
api_key = os.getenv('API_KEY')
base_url = os.getenv('BASE_URL')
url = os.getenv('KG_URL')
auth = (os.getenv('KG_AUTH'), os.getenv('KG_PASS'))
dsr1_params = json.loads(os.getenv('DS_R1', '{}'))
dsv3_params = json.loads(os.getenv('DS_V3', '{}'))
qwen_params = json.loads(os.getenv('QWEN_MAX', '{}'))
emb_params = json.loads(os.getenv('EMBEDDING_MODEL', '{}'))
model_v3 = ChatOpenAI(base_url=base_url, openai_api_key=api_key, **dsv3_params)
model_q = ChatOpenAI(base_url=base_url, openai_api_key=api_key, **qwen_params)
output_parser = JsonOutputParser()
str_output_parser = StrOutputParser()

# 全局变量跟踪已处理事件
processed_event_hashes = set()

def calculate_event_hash(event):
    """
    计算事件哈希值用于去重
    """
    try:
        # 创建事件的字符串表示，确保一致性
        event_str = json.dumps(event, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(event_str.encode('utf-8')).hexdigest()
    except Exception:
        # 如果JSON序列化失败，使用字符串表示
        return hashlib.md5(str(event).encode('utf-8')).hexdigest()

def is_duplicate_event(event):
    """
    检查是否为重复事件
    """
    event_hash = calculate_event_hash(event)
    if event_hash in processed_event_hashes:
        return True
    processed_event_hashes.add(event_hash)
    return False

def print_event_with_dedup(event):
    """
    打印事件（带去重功能）
    """
    if not is_duplicate_event(event):
        # 尝试从多个可能的位置提取thought和output
        thought = extract_field_from_event(event, "thought")
        output = extract_field_from_event(event, "output")
        
        print("事件思考:", thought)
        print("事件输出:", output)
        print("<event_start>", event, "<event_end>")
        return True
    return False

def extract_field_from_event(event, field_name):
    """
    从事件的复杂结构中提取指定字段
    """
    # 首先尝试直接获取
    if field_name in event:
        return event[field_name]
    
    # 尝试从msg.content中解析
    if "msg" in event and hasattr(event["msg"], "content"):
        content = event["msg"].content
        # 尝试解析JSON格式的content
        try:
            if content.startswith("{") and content.endswith("}"):
                parsed_content = json.loads(content)
                if field_name in parsed_content:
                    return parsed_content[field_name]
        except:
            pass
        
        # 尝试从文本中提取field_name
        if f"{field_name}:" in content:
            lines = content.split("\n")
            for line in lines:
                if line.strip().startswith(f"{field_name}:"):
                    return line.split(f"{field_name}:", 1)[1].strip()
    
    # 尝试从其他可能的嵌套结构中提取
    for key, value in event.items():
        if isinstance(value, dict) and field_name in value:
            return value[field_name]
        elif isinstance(value, str):
            try:
                parsed_value = json.loads(value)
                if isinstance(parsed_value, dict) and field_name in parsed_value:
                    return parsed_value[field_name]
            except:
                pass
    
    return "无"


def test(
    question: str,
    file_path: str,
    config: dict,
    preview_n: int,
    save_dir: str = None,
):
    final_result = None

    state_data = {
        "question": question,
        "messages": [HumanMessage(content=question)],
    }
    if file_path:
        state_data["data_path"] = file_path
    if preview_n:
        state_data["preview_n"] = preview_n
    if save_dir:
        state_data["save_dir"] = save_dir

    initial_state = BrickState(**state_data)
    graph = build_graph_with_interaction(interrupt_after=["env_checker","data_analyzer"],interrupt_before=[])
    initial_state_dict = initial_state.model_dump()
    
    # 只在这里包装一次
    events = graph.stream(initial_state_dict, config=config, stream_mode="values")

    for event in events:
        status = event.get("status")
        print_event_with_dedup(event)
        
    try:
        while True:
            if status in ["FINISHED"]:
                break
            
            if status == "AWAITING_CONFIRMATION":
                thought = event.get('thought', 'Did you forget to upload your data?')
                output = event.get('output', 'Did you forget to upload your data?')
                update_data_info = input(f"System thought: {thought},\n System output:{output} \n")
                graph.update_state(config=config,values={"update_data_info": update_data_info})
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    print_event_with_dedup(event)
            elif status == "Revise":
                thought = event.get('thought', 'Did you forget to upload your data?')
                output = event.get('data_repo', 'Can not find data_repo')
                update_data_repo = input(f"System thought: {thought},\n System output:{output} \n")
                graph.update_state(config=config,values={"update_data_repo": update_data_repo})
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    print_event_with_dedup(event)
            elif status == "ASK_USER":
                thought = event.get('thought', 'Did you forget to upload your data?')
                output = event.get('output', 'Can not find output')
                user_input = input(f"System thought: {thought},\n System output:{output} \n")
                user_update_detail = event.get('user_update_detail',[])
                user_update_detail.append(user_input)
                graph.update_state(config=config,values={"user_update_detail": user_update_detail})
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    print_event_with_dedup(event)
            elif status == "IMPROVE_CONFIRMATION":
                thought = event.get('thought', 'Did you forget to upload your data?')
                output = event.get('re_plan', 'Can not find re_plan')
                user_input = input(f"System thought: {thought},\n System output:{output} \n")
                update_instruction = event.get('update_instruction',[])
                update_instruction.append(user_input)
                graph.update_state(config=config,values={"update_instruction": update_instruction})
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    print_event_with_dedup(event)
            elif status == "CODE_CONFIRMATION":
                interaction = event.get('code', 'Can not find code')
                code_decision = input("Do you want to modify the code? (yes/no)").strip().lower()
                update_code = "Don't need to modify the current code." if code_decision == "no" else input("Input your update code information:")
                graph.update_state(config=config,values={"update_code": update_code})
                events = graph.stream(None, config, stream_mode="values")
                for event in events:
                    print_event_with_dedup(event)
            elif status in ["VALIDATED","NOT_FINISHED"]:
                events = graph.stream(None, config, stream_mode="values")
                events_list = list(events)
                
                if len(events_list) == 0:
                    status = "FINISHED"
                    break
                
                events = iter(events_list)
                
            else:
                break
            
            latest_status = None
            for event in events:
                print_event_with_dedup(event)
                latest_status = event.get("status")
                if latest_status in ["FINISHED"]:
                    status = latest_status
                    break
            
            if latest_status is not None:
                status = latest_status
            else:
                user_choice = input("继续(c)/退出(q): ").strip().lower()
                if user_choice == 'q':
                    status = "FINISHED"
                    break
            
            if status in ["FINISHED"]:
                break
        
        if status in ["FINISHED"]:
            final_result = event.get("final_result","Oops! Something went wrong.")
    except StopIteration:
        pass
    
    return final_result

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "22"}, "recursion_limit": 200}
    question = input("请输入你的问题：")
    save_dir = get_save_dir(question)
    result = test(question, file_path="/home/lyt/checker_finallap/files/GSE84133_GSM2230761_mouse1_modified.h5ad", config=config, preview_n=20, save_dir=save_dir)
    print("final result: ",result)

# /usr/bin/python /home/lyt/checker_finallap/run.py | tee test_event2.txt