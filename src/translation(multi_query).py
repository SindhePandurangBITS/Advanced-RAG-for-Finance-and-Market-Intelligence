# src/translation(multi_query).py

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def decompose_question(question, model_name="gpt-4", n_queries=3):
    """
    Decompose a complex question into sub-questions using an LLM.
    Returns a list of sub-questions.
    """
    template = f"""
    You are a helpful assistant that generates multiple sub-questions for an input question.
    Break down: {{question}}
    Output ({n_queries} queries):
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = prompt | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    return chain.invoke({"question": question})

if __name__ == "__main__":
    question = "How are companies across different domains and industries adopting the AI revolution?"
    sub_questions = decompose_question(question)
    print(sub_questions)
