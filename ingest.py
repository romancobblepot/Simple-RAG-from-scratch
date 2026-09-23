"""
Ingestion script for the RAG Nutritional Chatbot.

Reads the Human Nutrition (2020 Edition) PDF, splits pages into
sentence chunks, embeds them with sentence-transformers/all-mpnet-base-v2
(768 dimensions) and uploads everything to the public.chunks table in
Supabase, where the match_documents RPC performs hybrid retrieval.

Requires a .env file with:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import re
import uuid

import numpy as np
import pandas as pd
import pymupdf
import fitz
import requests
import torch
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from supabase import Client, create_client
from tqdm.auto import tqdm

env_file = find_dotenv(usecwd=True)
print(f"Targeting .env file at: {env_file}")
print(f"Loaded successfully? {load_dotenv(env_file, verbose=True, override=True)}")

doc_path = "/Users/adnaniqbalkantroo/Downloads/Human-Nutrition-2020-Edition-1598491699.pdf"
os.path.exists(doc_path)

source = "Human Nitition Book 2020-Edition"
DOC_ID = "nutrition-v1"
BATCH_INSERT = 200

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]


def text_formatter(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("\n", " ").strip()
    return text


def open_and_read_pdf(doc_path: str) -> list[dict]:
    doc = fitz.open(doc_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append(
            {
                "page_number": page_number - 41,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
            }
        )
    return pages_and_texts


pages_and_texts = open_and_read_pdf(doc_path)

nlp = English()
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]


num_sentence_chunks = 10


def split_sentence_list(input_list: list, slice_size: int) -> list[list[str]]:
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]


for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_sentence_list(item["sentences"], num_sentence_chunks)
    item["num_chunks"] = len(item["sentence_chunks"])

pages_and_chunks = []

for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r"\.([A-Z])", r". \1", joined_sentence_chunk)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split())
        chunk_dict["chunk_token_count"] = round(len(joined_sentence_chunk) / 4, 2)

        pages_and_chunks.append(chunk_dict)

semantic_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

min_token_count = 30
temp_df = pd.DataFrame(pages_and_chunks)
temp_chunks = temp_df[temp_df["chunk_token_count"] > min_token_count].to_dict(orient="records")

sentence_chunk_and_embedding_df = pd.DataFrame(temp_chunks)
sentence_chunk_and_embedding_df_path = "sentence_chunk_and_embedding.csv"
sentence_chunk_and_embedding_df.to_csv(sentence_chunk_and_embedding_df_path, index=False)
sentence_chunk_and_embedding_df = pd.read_csv("sentence_chunk_and_embedding.csv")
pages_and_chunks = sentence_chunk_and_embedding_df.to_dict(orient="records")


def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    inputs, metadata, vectors = [], [], []

    for chunk in pages_and_chunks:
        if not chunk["sentence_chunk"]:
            continue
        else:
            inputs.append(chunk["sentence_chunk"])
            metadata.append({"page": chunk["page_number"], "source": source})
            vectors.append(semantic_model.encode(chunk["sentence_chunk"]).tolist())

    rows = []
    for idx, (content, emb, meta) in enumerate(zip(inputs, vectors, metadata)):
        rows.append(
            {
                "doc_id": DOC_ID,
                "chunk_index": idx,
                "content": content,
                "metadata": meta,
                "embedding": emb,
            }
        )

    for j in tqdm(range(0, len(rows), BATCH_INSERT), desc="Uploading to Supabase"):
        sb.table("chunks").insert(rows[j : j + BATCH_INSERT]).execute()

    print("Done Successfully!")


if __name__ == "__main__":
    main()
