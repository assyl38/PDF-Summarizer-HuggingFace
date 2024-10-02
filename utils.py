# import libraries

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from pypdf import PdfReader

def process_text(text):
    # process the text and split it into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,  # overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # load a model from HuggingFace to generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # create a FAISS index from the text chunks using the embeddings
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def summarizer(pdf):
    
    pdf_reader = PdfReader(pdf)
    text = ""

    # extract text from each page of the PDF
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    knowledgeBase = process_text(text)

    # Define the query
    query = "Summarize the content of the uploaded PDF file"

    if query:
        docs = knowledgeBase.similarity_search(query)

        # Use an open-source model from Hugging Face, like falcon or GPT-Neo
        summarization_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", max_new_tokens=200)

        llm = HuggingFacePipeline(pipeline=summarization_pipeline)

        chain = load_qa_chain(llm, chain_type='stuff')

        response = chain.run(input_documents=docs, question=query)
        
        return response
