# src/indexing(hierarchical_raptor).py

import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Set model for summarization
model = ChatOpenAI(temperature=0, model="gpt-4.1-nano")  # Use a cheap LLM for summarization

def embed(texts, embd):
    """Embed a list of texts using a provided embedding model."""
    return np.array(embd.embed_documents(texts))

def global_cluster_embeddings(embeddings, dim=10, metric="cosine"):
    n_neighbors = max(2, int((len(embeddings) - 1) ** 0.5))
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

def get_optimal_clusters(embeddings, max_clusters=30, random_state=224):
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]

def GMM_cluster(embeddings, threshold=0.1, random_state=0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def perform_clustering(texts, embd, dim=10, threshold=0.1):
    if len(texts) <= 2:
        embeddings = embed(texts, embd)
        return pd.DataFrame({"text": texts, "embd": list(embeddings), "cluster": [[0]] * len(texts)})
    embeddings = embed(texts, embd)
    reduced = global_cluster_embeddings(embeddings, dim)
    cluster_labels, n_clusters = GMM_cluster(reduced, threshold)
    return pd.DataFrame({"text": texts, "embd": list(embeddings), "cluster": cluster_labels})

def summarize_cluster(texts, model, max_chunks=3, max_tokens=2000):
    from tiktoken import get_encoding
    enc = get_encoding("cl100k_base")
    texts = sorted(texts, key=len, reverse=True)[:max_chunks]
    total = 0
    selected = []
    for t in texts:
        t_len = len(enc.encode(t))
        if total + t_len > max_tokens:
            break
        selected.append(t)
        total += t_len
    context = "\n\n".join(selected)
    prompt = "Summarize the following documentation for knowledge retrieval:\n{context}"
    chain = ChatPromptTemplate.from_template(prompt) | model | StrOutputParser()
    return chain.invoke({"context": context})

def recursive_cluster_summarize(texts, embd, model, n_levels=2, level=1, results=None):
    """Hierarchical recursive clustering and summarization."""
    if results is None:
        results = {}
    df = perform_clustering(texts, embd)
    results[level] = df
    clusters = []
    for c in df['cluster']:
        clusters.extend(c)
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 and level < n_levels:
        for cluster in unique_clusters:
            cluster_texts = df[[cluster in cl for cl in df['cluster']]]['text'].tolist()
            summary = summarize_cluster(cluster_texts, model)
            # Store summaries for later use
            if "summary" not in df.columns:
                df["summary"] = ""
            df.loc[[cluster in cl for cl in df['cluster']], "summary"] = summary
            recursive_cluster_summarize([summary], embd, model, n_levels, level+1, results)
    return results

if __name__ == "__main__":
    # Assume you already have leaf_texts and embd from previous pipeline steps
    # Example: from src.Indexing(Semantic_Splitter) import semantic_split_docs
    # leaf_texts = semantic_split_docs(all_docs)
    # from langchain_openai import OpenAIEmbeddings
    # embd = OpenAIEmbeddings()
    leaf_texts = []  # <-- Replace with your split docs
    embd = None      # <-- Replace with your embedding model

    # Run hierarchical clustering & summarization
    results = recursive_cluster_summarize(leaf_texts, embd, model, n_levels=2)
    print("Hierarchical summarization completed.")

    # Flatten all leaf_texts and all unique summaries from all levels
    all_texts = leaf_texts.copy()
    for lvl, df in results.items():
        if "summary" in df.columns:
            all_texts.extend(df["summary"].tolist())
    print(f"Total chunks and summaries: {len(all_texts)}")
