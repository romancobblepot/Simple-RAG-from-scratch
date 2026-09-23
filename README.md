# NutriSage AI

I have connected the Vizuara RAG Pipeline Supabase.

I have built a functionaly RAG pipeline:

Here is my SQL Editor code on Supabase: --

-- Enable pgvector
create extension if not exists vector;

-- Table for textbook chunks
create table if not exists public.chunks (
  doc_id text not null,
  chunk_index int not null,
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(768)
);

-- Vector index for cosine similarity retrieval
create index if not exists idx_chunks_embedding
  on public.chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Optional: helps keyword-search operations
create index if not exists idx_chunks_content_fts
  on public.chunks
  using gin (to_tsvector('english', content));

-- Remove older versions of the RPC
drop function if exists public.match_documents(vector, integer);
drop function if exists public.match_documents(vector, integer, jsonb);
drop function if exists public.match_documents(vector, text, integer, jsonb);

-- Hybrid retrieval:
-- 70% semantic/vector similarity + 30% keyword relevance
create function public.match_documents(
  query_embedding vector(768),
  query_text text,
  match_count int default 5,
  filter jsonb default '{}'::jsonb
)
returns table (
  doc_id text,
  chunk_index int,
  content text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    c.doc_id,
    c.chunk_index,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks as c
  where filter = '{}'::jsonb or c.metadata @> filter
  order by
    (
      0.7 * (1 - (c.embedding <=> query_embedding))
      + 0.3 * ts_rank_cd(
          to_tsvector('english', c.content),
          websearch_to_tsquery('english', query_text),
          32
        )
    ) desc
  limit match_count;
$$;

I have a script called ingest.py which has already created chunks of embeddings in supabase.

I have following files:

rag-chat/app/api/chat/route.ts:

import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

export const runtime = "nodejs";
export const maxDuration = 60;

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !GROQ_API_KEY) {
  throw new Error(
    "Missing SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, or GROQ_API_KEY.",
  );
}

// Server-only client: never expose SUPABASE_SERVICE_ROLE_KEY to the browser.
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

// Groq's OpenAI-compatible endpoint lets us use the official OpenAI SDK.
const groq = new OpenAI({
  apiKey: GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

type RetrievedChunk = {
  content: string;
  metadata: { page?: number; source?: string } | null;
  similarity?: number;
};

let embedderPromise: Promise<(text: string) => Promise<number[]>> | undefined;

import { InferenceClient } from "@huggingface/inference";
const hf = new InferenceClient(process.env.HF_TOKEN);

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const message = typeof body.message === "string" ? body.message.trim() : "";

    if (!message) {
      return NextResponse.json({ error: "message is required" }, { status: 400 });
    }

    if (message.length > 2_000) {
      return NextResponse.json({ error: "message is too long" }, { status: 400 });
    }

    const queryEmbedding = await hf.featureExtraction({
        model: "sentence-transformers/all-mpnet-base-v2",
        inputs: message,
    });
    // Requires the match_chunks RPC shown below. The doc filter keeps this
    // chatbot grounded only in the book ingested by ingest.py.
    const { data, error } = await supabase.rpc("match_documents", {
        query_embedding: queryEmbedding,
        query_text: message,
        match_count: 4,
        }
    );

    if (error) throw error;

    const chunks = (data ?? []) as RetrievedChunk[];
    console.log(
    "Retrieved chunks:",
    chunks.map((chunk) => ({
        page: chunk.metadata?.page,
        similarity: chunk.similarity,
        content: chunk.content,
    })),
    );
    if (!chunks.length) {
      return NextResponse.json({
        answer: "I couldn't find relevant information in the nutrition book.",
        sources: [],
      });
    }

    // Keep Groq input modest; long full chunks are unnecessary and invite limits.
    const context = chunks
    .map((chunk) => {
        const source = chunk.metadata?.source ?? "";
        const page = chunk.metadata?.page ?? "";

        return `[${source}, page ${page}]\n${chunk.content.slice(0, 3_500)}`;
    })
    .join("\n\n");

    const completion = await groq.chat.completions.create({
      model: "qwen/qwen3.8-27b",
      temperature: 0.2,
      max_tokens: 600,
      messages: [
        {
          role: "system",
          content:
            "You are a helpful nutrition-book assistant. Answer only from the provided context. If the context does not contain the answer, say so plainly. Do not reveal reasoning or invent facts. This is educational information, not medical diagnosis."
        },
        {
          role: "user",
          content: `Context:\n${context}\n\nQuestion: ${message}`,
        },
      ],
    });

    const rawAnswer =
    completion.choices[0]?.message?.content?.trim() ??
    "I couldn't generate an answer.";

    const citations = [
    ...new Set(
        chunks.map((chunk) => {
        const source = chunk.metadata?.source ?? "Human Nutrition Book";
        const page = chunk.metadata?.page;
        return page === undefined ? source : `${source}, page ${page}`;
        }),
    ),
    ];

    const answer = `${rawAnswer}\n\nSources: ${citations
    .map((citation) => `[${citation}]`)
    .join(" ")}`;

    return NextResponse.json({
      answer: answer || "I couldn't find this in the provided documents. Please try rephrasing in a better way.",
    });
  } catch (error) {
    console.error("Chat route error:", error);
    return NextResponse.json(
      { error: "Unable to answer the question right now." },
      { status: 500 },
    );
  }
}

ingest.py:

import torch

import os,uuid,re

from dotenv import load_dotenv,find_dotenv

import requests

import pymupdf

import fitz

import pandas as pd

import numpy as np

from tqdm.auto import tqdm

from supabase import create_client,Client

from pathlib import Path

from spacy.lang.en import English

from sentence_transformers import SentenceTransformer,util

import nltk






env_file = find_dotenv(usecwd=True)

print(f"Targeting .env file at: {env_file}")

print(f"Loaded successfully? {load_dotenv(env_file, verbose=True,override=True)}")




doc_path="/Users/adnaniqbalkantroo/Downloads/Human-Nutrition-2020-Edition-1598491699.pdf"

os.path.exists(doc_path)

source="Human Nitition Book 2020-Edition"

DOC_ID="nutrition-v1"

BATCH_INSERT=200

SUPABASE_URL=os.environ['SUPABASE_URL']

SUPABASE_SERVICE_ROLE_KEY=os.environ['SUPABASE_SERVICE_ROLE_KEY']





def text_formatter(text:str) ->str:

text=text.replace("\r"," ")

text=re.sub(r"-\s*\n\s*","",text)

text=re.sub(r"\s+\n","\n",text)

text=re.sub(r"[ \t]+"," ",text)

text=text.replace("\n"," ").strip()

return text




def open_and_read_pdf(doc_path:str)->list[dict]:

doc=fitz.open(doc_path)

pages_and_texts=[]

for page_number,page in tqdm(enumerate(doc)):

text=page.get_text()

text=text_formatter(text)

pages_and_texts.append({

"page_number":page_number-41,

"page_char_count":len(text),

"page_word_count":len(text.split(" ")),

"page_sentence_count_raw":len(text.split(". ")),

"page_token_count":len(text)/4,

"text":text

                                }

                              )

return pages_and_texts




pages_and_texts=open_and_read_pdf(doc_path)




nlp=English()

nlp.add_pipe('sentencizer')

for item in tqdm(pages_and_texts):

item["sentences"]=list(nlp(item["text"]).sents)

item["sentences"]=[str(sentence) for sentence in item["sentences"]]




num_sentence_chunks=10

def split_sentence_list(input_list: list, slice_size: int) ->list[list[str]]:

return [input_list[i:i+slice_size] for i in range(0,len(input_list),slice_size)]




for item in tqdm(pages_and_texts):

item['sentence_chunks']=split_sentence_list(item['sentences'],num_sentence_chunks)

item["num_chunks"]=len(item['sentence_chunks'])




pages_and_chunks=[]




for item in tqdm(pages_and_texts):

for sentence_chunk in item["sentence_chunks"]:

chunk_dict={}

chunk_dict["page_number"]=item["page_number"]




joined_sentence_chunk="".join(sentence_chunk).replace("  "," ").strip()

joined_sentence_chunk=re.sub(r'\.([A-Z])', r'. \1',joined_sentence_chunk)

chunk_dict["sentence_chunk"]=joined_sentence_chunk

chunk_dict["chunk_char_count"]=len(joined_sentence_chunk)

chunk_dict["chunk_word_count"]=len(joined_sentence_chunk.split())

chunk_dict["chunk_token_count"]=round(len(joined_sentence_chunk)/4,2)




pages_and_chunks.append(chunk_dict)





semantic_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")




min_token_count=30

temp_df=pd.DataFrame(pages_and_chunks)

temp_chunks=temp_df[temp_df['chunk_token_count']>min_token_count].to_dict(orient='records')




sentence_chunk_and_embedding_df=pd.DataFrame(temp_chunks)

sentence_chunk_and_embedding_df_path="sentence_chunk_and_embedding.csv"

sentence_chunk_and_embedding_df.to_csv(sentence_chunk_and_embedding_df_path,index=False)

sentence_chunk_and_embedding_df=pd.read_csv('sentence_chunk_and_embedding.csv')

pages_and_chunks=sentence_chunk_and_embedding_df.to_dict(orient="records")




def main():

sb: Client=create_client(SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY)





inputs,metadata,vectors=[],[],[]

for chunk in pages_and_chunks:

if not chunk['sentence_chunk']:

continue

else:

inputs.append(chunk['sentence_chunk'])

metadata.append({'page':chunk['page_number'],'source':source})

vectors.append(semantic_model.encode(chunk['sentence_chunk']).tolist())




rows=[]

for idx,(content,emb,meta) in enumerate(zip(inputs,vectors,metadata)):

rows.append({

"doc_id":DOC_ID,

"chunk_index":idx,

"content":content,

"metadata":meta,

"embedding":emb,




        })




for j in tqdm(range(0,len(rows),BATCH_INSERT),desc="Uploading to Supabase"):

sb.table("chunks").insert(rows[j:j+BATCH_INSERT]).execute()




print("Done Successfully!")




if __name__=="__main__":

main()



rag-chat/app/page.tsx:

"use client";




import { FormEvent, useState } from "react";




type Source = {

page: number | null;

source: string | null;

similarity: number | null;

};




type ChatMessage = {

role: "user" | "assistant";

content: string;

sources?: Source[];

};




const starterQuestions = [

"How often should infants be breastfed?",

"What are the fat-soluble vitamins?",

"What are the symptoms of pellagra?",

];




export default function Home() {

const [messages, setMessages] = useState<ChatMessage[]>([]);

const [input, setInput] = useState("");

const [isLoading, setIsLoading] = useState(false);

const [error, setError] = useState("");




async function sendMessage(event?: FormEvent, preset?: string) {

event?.preventDefault();

const message = (preset ?? input).trim();

if (!message || isLoading) return;




setError("");

setInput("");

setMessages((current) => [...current, { role: "user", content: message }]);

setIsLoading(true);




try {

const response = await fetch("/api/chat", {

method: "POST",

headers: { "Content-Type": "application/json" },

body: JSON.stringify({ message }),

      });




const data = await response.json();

if (!response.ok) throw new Error(data.error || "Unable to get an answer.");




setMessages((current) => [

...current,

        { role: "assistant", content: data.answer, sources: data.sources },

      ]);

    } catch (err) {

setError(err instanceof Error ? err.message : "Something went wrong.");

    } finally {

setIsLoading(false);

    }

  }




return (

<main className="min-h-screen bg-stone-50 text-stone-900">

<div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-8 sm:px-6">

<header className="mb-8 text-center">

<p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-emerald-700">

            Retrieval-augmented nutrition assistant

</p>

<h1 className="text-3xl font-bold tracking-tight sm:text-4xl">

            Ask the Human Nutrition book

</h1>

<p className="mx-auto mt-3 max-w-2xl text-sm leading-6 text-stone-600">

            Answers are grounded in the 2020 Human Nutrition textbook. This is

            educational information, not personal medical advice.

</p>

</header>




<section className="flex flex-1 flex-col rounded-2xl border border-stone-200 bg-white shadow-sm">

<div className="flex-1 space-y-5 overflow-y-auto p-5 sm:p-7">

{messages.length === 0 && (

<div className="py-10 text-center">

<p className="text-stone-600">Try one of these questions:</p>

<div className="mt-5 flex flex-wrap justify-center gap-2">

{starterQuestions.map((question) => (

<button

key={question}

onClick={() => sendMessage(undefined, question)}

className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-900 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"

disabled={isLoading}

>

{question}

</button>

                  ))}

</div>

</div>

            )}




{messages.map((message, index) => (

<article

key={`${message.role}-${index}`}

className={message.role === "user" ? "ml-auto max-w-[85%]" : "max-w-[90%]"}

>

<div

className={

message.role === "user"

? "rounded-2xl rounded-br-md bg-emerald-700 px-4 py-3 text-white"

: "rounded-2xl rounded-bl-md bg-stone-100 px-4 py-3 text-stone-800"

}

>

<p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>

</div>




{message.role === "assistant" && message.sources?.length ? (

<p className="mt-2 text-xs text-stone-500">

                    Sources: {message.sources

                      .map((source) =>

source.page === null ? source.source || "Nutrition book" : `page ${source.page}`,

                      )

                      .filter((value, index, values) => values.indexOf(value) === index)

                      .join(", ")}

</p>

                ) : null}

</article>

            ))}




{isLoading && (

<div className="max-w-[90%] rounded-2xl rounded-bl-md bg-stone-100 px-4 py-3 text-sm text-stone-500">

                Searching the book and writing an answer…

</div>

            )}

</div>




<form onSubmit={(event) => sendMessage(event)} className="border-t border-stone-200 p-4">

{error && <p className="mb-3 text-sm text-red-600">{error}</p>}

<div className="flex gap-2">

<label htmlFor="question" className="sr-only">

                Ask a nutrition question

</label>

<input

id="question"

value={input}

onChange={(event) => setInput(event.target.value)}

placeholder="Ask a question about nutrition…"

disabled={isLoading}

className="min-w-0 flex-1 rounded-xl border border-stone-300 px-4 py-3 text-sm outline-none transition placeholder:text-stone-400 focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100 disabled:bg-stone-50"

/>

<button

type="submit"

disabled={!input.trim() || isLoading}

className="rounded-xl bg-emerald-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"

>

                Send

</button>

</div>

</form>

</section>

</div>

</main>

  );

}

Deploy the full RAG application for me. end to end using the above. Here are my .env credentials:

HF_TOKEN=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
GROQ_API_KEY='@secret:GROQ_API_KEY '
OPENAI_API_KEY='@secret:OPENAI_API_KEY '
SUPABASE_URL='https://glsoelsddngxesycvlux.supabase.co'
SUPABASE_SERVICE_ROLE_KEY='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdsc29lbHNkZG5neGVzeWN2bHV4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MDAwOTM4MSwiZXhwIjoyMTA1NTg1MzgxfQ.jDM9-zQwBMNp-VFjUNlnX9JJBSZiaslVZTbwjQ8Uems'

You will already have my SUPABASE KEYS SINCE I HAVE CONNECTED MY LOVABLE ACCOUNT

Make the look of the website very cool!! The chat interface should look minimal and sleek like OpenAI





Show three dots when responding. 




Show citations when you have pages to cite. I mean clickable citations, when I click on it a small popup highlighting the place from the text it was taken should be shown!




Instead of Nutriton AI: Can you give the name as RAG Nutritional Chatbot: Build from Scratch

Made by Adnan Iqbal Kantroo(romancobblepot)

Make theme of text and font etc as this website:

This theme is very, very important to follow: https://arcprize.org/arc-agi

Also show small sounds when the three dots are coming and small sound when response is generated

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://wisdom-well-nutri.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/c80dbb66-545e-4b51-b36d-385a8f096187).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
