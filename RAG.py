import gradio as gr
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# PDF processing and RAG pipeline setup function
def setup_rag_pipeline(pdf_path):
    """
    Sets up a RetrievalQA pipeline using a PyPDFLoader to load and split a PDF file,
    and a HuggingFacePipeline for text generation.

    Parameters:
        pdf_path (str): The path to the PDF file to load and process.

    Returns:
        RetrievalQA: A RetrievalQA chain instance with a contextual compression retriever.
    """

    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load_and_split()
    model_id = "codegood/Llama_3.1_8B_GGUF"
    file_name= "meta-llama-3.1-8b-instruct-q2_k.gguf"
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file = file_name,
                                              trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file = file_name,
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)  
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024,
                    device="auto")
    llm = HuggingFacePipeline(pipeline=pipe)
    reranker = HuggingFaceCrossEncoder(model_name="mixedbread-ai/mxbai-rerank-large-v1", model_kwargs = {'device': device})# Reranker Model

    embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1",
                                model_kwargs={"device":device}, encode_kwargs = {"normalize_embeddings":True}) # Embedding Model
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents=documents)

    vectordb = FAISS.from_documents(documents=texts, embedding=embed_model)
    retriever = vectordb.as_retriever(k=len(documents)//2, threshold=0.10)

    # Optionally save the vector index
    vectordb.save_local("vectorDB")

    compressor = CrossEncoderReranker(model=reranker, top_n=len(documents)//3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    prompt = """
        You are an expert AI providing accurate and concise answers based on the following information:

        {context}

        Question: {question}

        Provide a clear and informative response. Don't include any irrelevant information AND Don't repeat.
        """
    prompt_template = PromptTemplate(template=prompt, input_variables=["context"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )
    return qa_chain

file_path = input("Enter the file path: ")
qa_chain = setup_rag_pipeline(file_path)


if __name__ == '__main__':
    while True:
        question = input("Enter the question: ")
        response = qa_chain.invoke(question)
        print(response['result'])
