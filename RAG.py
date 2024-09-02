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
from langchain_ollama import OllamaEmbeddings

def setup_rag_pipeline(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load_and_split()

    embed_model = OllamaEmbeddings(model="mxbai-embed-large", num_gpu=1,
                                    model_kwargs={"normalize":True})  # Embedding Model
    llm = OllamaLLM(model="llama3.1", num_gpu=1, num_thread=8, repeat_penalty=1.1, 
                    num_beams = 3, top_p = 0.95, top_k = 20, num_predict=3056)  # Language Model
    reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")  # Reranker Model

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=180)
    texts = text_splitter.split_documents(documents=documents)

    vectordb = FAISS.from_documents(documents=texts, embedding=embed_model)
    retriever = vectordb.as_retriever(k=20)

    # Optionally save the vector index
    vectordb.save_local("medical_faiss_index")

    compressor = CrossEncoderReranker(model=reranker, top_n=4)
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
# Function to answer questions based on the PDF
def answer_question(pdf, chat_history, question):
    qa_chain = setup_rag_pipeline(pdf.name)
    response = qa_chain.invoke(question)
    answer = response['result']
    chat_history.append((question, answer))
    return chat_history, chat_history

# Create the Gradio interface
with gr.Blocks(css=".chatbox .textbox {min-height: 80px; }") as demo:
    with gr.Row():
        gr.Markdown("# Local RAG Chatbot")

    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])

    with gr.Column():
        chatbot = gr.Chatbot(label="Chat History")
        question_input = gr.Textbox(label="Ask a Question", placeholder="What are the benefits of Anthem Gold PPO?")

    submit_btn = gr.Button("Send")
    chat_history = gr.State([])

    submit_btn.click(fn=answer_question, inputs=[pdf_input, chat_history, question_input], outputs=[chatbot, chat_history])
    question_input.submit(fn=answer_question, inputs=[pdf_input, chat_history, question_input], outputs=[chatbot, chat_history])

# Launch the Gradio app
demo.launch()
