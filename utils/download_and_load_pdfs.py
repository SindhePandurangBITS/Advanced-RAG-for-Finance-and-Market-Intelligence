# src/utils/download_and_load_pdfs.py

import requests
from langchain_community.document_loaders import PyPDFLoader

def download_and_load_pdfs(pdf_urls):
    """
    Downloads PDF files from given URLs, loads them with PyPDFLoader,
    and returns a list of LangChain Documents with metadata.
    """
    all_docs = []
    for url, report_name in pdf_urls:
        try:
            filename = f"{report_name}.pdf"
            print(f"Downloading: {report_name} ...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            loader = PyPDFLoader(filename)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = report_name
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {report_name}")
        except Exception as e:
            print(f"Failed for {report_name}: {e}")
    return all_docs

if __name__ == "__main__":
    pdf_urls = [
        ("https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/2025/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf", "mckinsey_ai_report"),
        ("https://media-publications.bcg.com/BCG-Wheres-the-Value-in-AI.pdf", "bcg_ai_report"),
        ("https://www.citiwarrants.com/home/upload/citi_research/rsch_pdf_30305836.pdf", "citigroup_ai_report")
    ]
    all_docs = download_and_load_pdfs(pdf_urls)
    print(f"Total loaded pages: {len(all_docs)}")
