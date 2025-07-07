from flask import Flask, request, jsonify
from langsmith import Client
import pinecone
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter
from typing import List
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 从环境变量读取密钥
openai_key = os.environ.get("OPENAI_API_KEY")
pinecone_key = os.environ.get("PINECONE_API_KEY")
langsmith_key = os.environ.get("LANGSMITH_API_KEY")

# 初始化 API 客户端
client = Client(api_key=langsmith_key)
embeddings = OpenAIEmbeddings(api_key=openai_key)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
pinecone.init(api_key=pinecone_key, environment="us-east-1-aws")
index = pinecone.Index(index_name)

# 向量搜索初始化
index_name = "unlock-your-potential"
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.85},
)

# Prompt模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """RESPONSE GUIDELINES:
    - If a specific answer format is requested, follow it exactly
    - Otherwise, respond in a concise, supportive manner (3-5 sentences)
    - Use empathetic, warm, and non-judgmental tone
    ...
    DATABASE INFORMATION:
    {context}
    """),
    ("human", "{question}"),
])

# 文本格式化函数
def format_docs(documents: List[Document]) -> str:
    return "\n\n".join(f"Page Content: {doc.page_content}" for doc in documents)

# 构建处理链
format_doc = itemgetter("docs") | RunnableLambda(format_docs)
answer = prompt | llm | StrOutputParser()

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format_doc)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

# 路由
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message")

        if not message:
            return jsonify({"error": "Missing 'message' in request."}), 400

        result = chain.invoke(message)
        return jsonify({"response": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 启动服务
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render 会传入 PORT
    app.run(host="0.0.0.0", port=port)
