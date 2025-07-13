# src/retrieval(crag).py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict

# --- Data Model for Grading ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

llm_grader = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. Give a binary score 'yes' or 'no'."),
    ("human", "Retrieved document:\n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grade_prompt | llm_grader

# --- RAG Generation Chain ---
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

# --- Question Rewriter ---
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a question re-writer that converts an input question to a better version that is optimized for web search. Look at the input and try to reason about the underlying semantic intent / meaning."),
    ("human", "Here is the initial question:\n\n {question} \n Formulate an improved question."),
])
question_rewriter = re_write_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]

def retrieve(state):
    docs = retriever.get_relevant_documents(state["question"])
    return {"documents": docs, "question": state["question"]}

def grade_documents(state):
    question, documents = state["question"], state["documents"]
    filtered_docs, web_search = [], "No"
    for d in documents:
        grade = retrieval_grader.invoke({"question": question, "document": d.page_content}).binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate(state):
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    generation = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}

def transform_query(state):
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}

def web_search(state):
    docs = web_search_tool.invoke({"query": state["question"]})
    web_results = Document(page_content="\n".join([d["content"] for d in docs]))
    return {"documents": state["documents"] + [web_results], "question": state["question"]}

def decide_to_generate(state):
    return "transform_query" if state["web_search"] == "Yes" else "generate"

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "transform_query": "transform_query",
    "generate": "generate",
})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

if __name__ == "__main__":
    from pprint import pprint
    # Make sure to define your own retriever before running!
    # Example: from src.indexing(vectorize_chroma) import build_chroma_vectorstore
    # retriever = build_chroma_vectorstore(all_texts, embd)
    retriever = None  # <-- Replace with your retriever!

    inputs = {"question": "In what ways can AI-powered predictive analytics enhance investment strategies?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")
    pprint(value["generation"])
