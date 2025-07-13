# src/routing(semantic).py

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.utils.math import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Define domain-specific prompts (add more domains as needed)
domain_templates = [
    """You are an expert on AI adoption in technology companies.
    Provide practical insights and examples of how tech industry leaders use AI, including specific technologies and impact.
    Here is a question:
    {query}""",
    """You are an expert on AI in manufacturing and traditional industries.
    Explain how companies in manufacturing or healthcare use AI, with focus on challenges and integration processes.
    Here is a question:
    {query}"""
]
embeddings = OpenAIEmbeddings()
template_embeddings = embeddings.embed_documents(domain_templates)

def semantic_prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], template_embeddings)[0]
    idx = similarity.argmax()
    selected_template = domain_templates[idx]
    return ChatPromptTemplate.from_template(selected_template)

def route_and_generate(sub_questions, llm):
    answers = []
    for q in sub_questions:
        chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(semantic_prompt_router)
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(q)
        answers.append({"question": q, "answer": answer})
    return answers

if __name__ == "__main__":
    sub_questions = [
        "What specific AI technologies are companies in the tech industry implementing compared to companies in traditional industries like manufacturing or healthcare?",
        "How are companies in different domains approaching the integration of AI into their existing processes and systems?",
        "What challenges are companies facing in adopting AI across various industries, and how are they overcoming these obstacles?"
    ]
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    final_answers = route_and_generate(sub_questions, llm)
    for qa in final_answers:
        print(f"\nQuestion: {qa['question']}\nAnswer: {qa['answer']}\n")
