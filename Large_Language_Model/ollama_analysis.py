from ollama import chat, ChatResponse
import os

os.environ['OLLAMA_GPU_METAL'] = '10'
# 读取本地文件内容
def read_analysis_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# 构建分析提示
def create_analysis_prompt(file_content):
    prompt = f"""
    请严格按照以下要求分析宏病毒样本数据：

    【数据开始】
    {file_content}
    【数据结束】

    【分析要求】
    1. 行间比较，具有相同Species但不同Genotype或Isolate的认为重复行,选取其中一行作为代表,选取标准如下：
       - Avg_fold(平均覆盖深度)：越高表示该物种丰度越高
       - Covered_percent(覆盖百分比)：越高表示序列完整性越好
       - Covered_bases(覆盖碱基数):越高越好
       - Median_fold(中位覆盖深度):越高越好
       - Plus_reads/Minus_reads:比对到Species正负链reads数
    
    2. 输出格式：严格遵循如下表格形式：
    |物种名称|Avg_fold |Covered_percent | Median_fold｜Plus_reads | Minus_reads |
    
    3. 禁止输出思考过程
    请直接给出分析结果：
    """
    print(prompt)
    return prompt

# 主分析函数
def analyze_metagenomics_sample(file_path):
    # 读取文件内容
    file_content = read_analysis_file(file_path)

    # 创建分析提示
    prompt = create_analysis_prompt(file_content)

    # 使用Ollama进行聊天分析
    response: ChatResponse = chat(
        model='deepseek-r1:latest',  # 或者你本地安装的其他模型，如 'mistral', 'qwen' 等
        messages=[
            {
                'role': 'user',
                'system': '直接回答，不要思考过程',
                'content': prompt
            }
        ],
        options = {
        'temperature': 0.0,  # Set to 0.0 for maximum determinism
        'seed': 42  # Use a specific integer seed (e.g., 42, 123)
        }
    )

    return response['message']['content']

# 使用示例
if __name__ == "__main__":
    file_path = "./test.txt"  # 你的文件路径
    result = analyze_metagenomics_sample(file_path)
    print("分析结果：")
    print(result)