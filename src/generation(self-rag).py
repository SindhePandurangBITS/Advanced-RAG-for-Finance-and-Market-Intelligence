# src/generation(self-rag).py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from pprint import pprint

# --------- Data Models for Grading ---------
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# --------- Prompts and Chains ---------
llm_gpt4o = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords or semantic meaning related to the question, grade it as relevant. Give a binary score 'yes' or 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grade_prompt | llm_gpt4o.with_structured_output(GradeDocuments)

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'."),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])
hallucination_grader = hallucination_prompt | llm_gpt4o.with_structured_output(GradeHallucinations)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer addresses / resolves a question. Give a binary score 'yes' or 'no'."),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])
answer_grader = answer_prompt | llm_gpt4o.with_structured_output(GradeAnswer)

prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm_gpt35 | StrOutputParser()

# --------- Graph State Definition ---------
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --------- Pipeline Nodes ---------
def retrieve(state):
    # Assumes retriever is already defined elsewhere and injected
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    question, documents = state["question"], state["documents"]
    filtered_docs = [
        d for d in documents
        if retrieval_grader.invoke({"question": question, "document": d.page_content}).binary_score == "yes"
    ]
    return {"documents": filtered_docs, "question": question}

def generate(state):
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    generation = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}

def decide_to_generate(state):
    return "generate" if state["documents"] else "retrieve"

def grade_generation_v_documents_and_question(state):
    question, documents, generation = state["question"], state["documents"], state["generation"]
    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation}).binary_score
    if hallucination_score == "yes":
        answer_score = answer_grader.invoke({"question": question, "generation": generation}).binary_score
        return "useful" if answer_score == "yes" else "not useful"
    else:
        return "not supported"

# --------- Build and Run the Workflow Graph ---------
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "generate": "generate",
    "retrieve": "retrieve",
})
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "retrieve",
        "not supported": "generate",
    },
)
app = workflow.compile()

if __name__ == "__main__":
    # Make sure you have a retriever defined and injected before running!
    retriever = None  # <-- Replace with your retriever!
    inputs = {"question": "How can AI be used to improve customer personalization in financial services?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")
    pprint(value["generation"])
