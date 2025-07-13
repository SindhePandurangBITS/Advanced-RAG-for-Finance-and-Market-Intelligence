# src/retrieval_generation(multi_query).py

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def decompose_question(question, model_name="gpt-4", n_queries=3):
    """Decompose a complex question into sub-questions using an LLM."""
    template = f"""
    You are a helpful assistant that generates multiple sub-questions for an input question.
    Break down: {{question}}
    Output ({n_queries} queries):
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = prompt | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    return chain.invoke({"question": question})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_and_generate(sub_questions, retriever, llm):
    """Retrieve docs and generate answers for each sub-question."""
    rag_template = """
    Context:
    {context}

    Question:
    {question}

    Answer concisely:
    """
    prompt_rag = ChatPromptTemplate.from_template(rag_template)
    answers = []
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        context = format_docs(docs)
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": context, "question": sq})
        answers.append(answer)
    return answers

def synthesize_final_answer(sub_questions, answers, main_question, llm):
    """Aggregate all Q&A pairs into a final answer using LLM."""
    def format_qa_pairs(questions, answers):
        return "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)])
    context = format_qa_pairs(sub_questions, answers)
    final_template = """
    Here are Q&A pairs:

    {context}

    Use these to write a comprehensive answer to the question:
    {question}
    """
    final_prompt = ChatPromptTemplate.from_template(final_template)
    return (final_prompt | llm | StrOutputParser()).invoke({"context": context, "question": main_question})

if __name__ == "__main__":
    # Example usage: assumes retriever is already defined/imported elsewhere
    question = "How are companies across different domains and industries adopting the AI revolution?"
    retriever = None  # <-- Replace with your retriever instance!
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    sub_questions = decompose_question(question)
    answers = retrieve_and_generate(sub_questions, retriever, llm)
    final_answer = synthesize_final_answer(sub_questions, answers, question, llm)

    print("Sub-questions:", sub_questions)
    print("\nFinal answer:\n", final_answer)


