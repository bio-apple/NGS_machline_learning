## 本地运行大语言模型

*	硬件要求

        CPU：多核处理器（推荐 4 核或以上）。
        GPU：如果你计划运行大型模型或进行微调，推荐使用具有较高计算能力的 GPU（如 NVIDIA 的 CUDA 支持）。
        内存：至少 8GB RAM，运行较大模型时推荐 16GB 或更高。
        存储：需要足够的硬盘空间来存储预训练模型，通常需要 10GB 至数百 GB 的空间，具体取决于模型的大小。
        软件要求：确保系统上安装了最新版本的 Python（如果打算使用 Python SDK）

**You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.**

*	个人电脑(最好是Mac可以无缝使用GPU加速,windows也可以)

*   安装好python3

*   电脑安装[Ollama](https://ollama.com)

*   启动Ollama

        ollama serve

*   安装python模块

        pip install ollama

*   [模型下载](https://ollama.com/models),本地安装下载，示例代码：

        ollama run deepseek-r1

    可用的开源模型

    | Model               | Parameters(latest,max) | Size         | Download                                                                                           |
    |---------------------|------------------------|--------------|----------------------------------------------------------------------------------------------------|
    | **Gemma 3(Google)** | 4B,27B                 | 3.3GB,17GB   |`ollama run gemma3`<br>`ollama run gemma3:27b` |
    | **Qwen(Alibaba Cloud)**          | 4B，110B                | 2.3GB，63GB   | `ollama run qwen:4b`<br>`ollama run qwen:110b`                                                                                |
    | **DeepSeek-R1**     | 8B, 671B               | 4.7GB, 404GB | `ollama run deepseek-r1`<br>`ollama run deepseek-r1:671b`                                          |
    | **Llama 4(Meta)**   | 109B, 400B             | 67GB, 245GB  | `ollama run llama4:scout`<br>`ollama run llama4:maverick`                                          |
    | **Phi 4(Microsoft)** | 3.8B, 14B              | 2.5GB, 9.1GB | `ollama run phi4-mini`<br>`ollama run phi4`                                                        |
