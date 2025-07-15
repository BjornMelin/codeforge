from langchain_openai import ChatOpenAI

# Define LLMs with different models via OpenRouter
planner_llm = ChatOpenAI(model_name="openai/o3", temperature=0.0)
coder_llm = ChatOpenAI(model_name="moonshotai/kimi-k2", temperature=0.1)
researcher_llm = ChatOpenAI(
    model_name="google/gemini-2.5-flash-preview-05-20", temperature=0.1
)
