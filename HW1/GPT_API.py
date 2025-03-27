from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken
from groq import Groq

load_dotenv()
OpenAI_api_key = os.getenv("GPT_API")
groq_api_key = os.getenv("GROQ_API")

client = OpenAI(
    api_key=OpenAI_api_key
)

groq_client = Groq(
    api_key=groq_api_key
)

tk = tiktoken.encoding_for_model("gpt-4o-mini")

def gen_prompt( role ):
    system_prompt = ""
    if ( role == "programing" ):
        system_prompt = """You are a senior software engineer and AI researcher with 10+ years of experience. 
        You specialize in Python, C++, JavaScript, and system architecture. 
        Your role is to provide precise, well-explained, and best-practice programming solutions. 
        Whenever possible, include code snippets, real-world applications, and performance optimizations. 
        Use a professional yet friendly tone when explaining concepts.
        """

    elif ( role == "writer" ):
        system_prompt = """You are a highly skilled fiction writer who follows the classic Three-Act Structure to craft engaging stories. 
        Each story should include:
        1. A gripping hook in the introduction.
        2. A well-paced middle with rising tension and character development.
        3. A satisfying climax and resolution.
        Your writing should be clear, immersive, and character-driven.
        """
    
    elif ( role == "emotional" ):
        system_prompt = """You are the ultimate hype master, a grandmaster of emotional support who delivers over-the-top praise and encouragement with a touch of humor.
        You speak like a mix of a motivational coach, a stand-up comedian, and a royal announcer.
        Your goal is to make people feel **legendary**, **unstoppable**, and **utterly amazing**—even if they just got out of bed.
        Use exaggerated metaphors, unexpected compliments, and a humorous, theatrical tone to lift their spirits.
        """
    
    else:
        raise ValueError("Invalid role")
    
    return [{"role": "system", "content": system_prompt}]
        
def truncate(messages, limit=300):
    total = 0
    new_messages = list()
    for msg in reversed(messages[1:]):
        total += len(tk.encode(msg["content"]))
        if total > limit:
            break
        new_messages.insert(0, msg)
    new_messages.insert(0, messages[0])
    return new_messages

def chat(messages, mode):
    if mode == "chat":
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
    elif mode == "special":
        return groq_client.chat.completions.create(
            messages=messages,
            model="mistral-saba-24b",
            stream=True
        )
    else:
        raise ValueError("Invalid mode")
    
def UI():
    print("歡迎使用 GPT-4o-mini 聊天機器人！")
    print("輸入 exit 或 quit 來退出聊天。")
    role = "init"
    while role == "init":    
        print("請選擇你要聊天的對象( 1.👨‍💻程式專家💻 2.📖科幻作家🚀 3.😊 情緒價值專家💖)：", end="")
        role = input().strip()
        if role == "1":
            role = "programing"
        elif role == "2":
            role = "writer"
        elif role == "3":
            role = "emotional"
        else:
            print("請輸入1~3的選項")
            role = "init"
    return role
    
def answer( msg, mode ):
    if mode == "chat":
        print(end="Assistant: ", flush=True)
    elif mode == "special":
        print(end=f"Assistant(special): ", flush=True)
    
    full_resp = str()
    for resp in msg:
        if not resp.choices:
            continue
        token = resp.choices[0].delta.content
        if token:
            print(token, end="", flush=True)
            full_resp += token
    print()
    return full_resp
    
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')    

def main():
    clear_terminal()
    mode = "chat"
    try:
        role = None
        role = UI()
        messages = gen_prompt(role)
        
        while True:
            prompt = input("User: ").strip()
            if prompt.lower() in ["exit", "quit"]:
                messages.append({"role": "user", "content": "我要退出對話"})

            elif prompt == "clear" and mode == "special":
                clear_terminal()
                messages = gen_prompt("special")
                print("已刷新對話")
                continue
            else:
                messages.append({"role": "user", "content": prompt})
                
            messages = truncate(messages)
            response = chat(messages, mode)
            full_resp = answer(response, mode)
            messages.append({"role": "assistant", "content": full_resp})
            
            if prompt.lower() in ["exit", "quit"]:
                break
            
    except KeyboardInterrupt:
        if role is None:
            print("\n你悄悄的離開了對話")
        else:
            messages = gen_prompt(role)
            messages.append({"role": "user", "content": "我偷偷的離開對話"})
            messages = truncate(messages)
            response = chat(messages, mode)
            answer(response, mode)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()