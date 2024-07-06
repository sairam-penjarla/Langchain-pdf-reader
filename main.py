# Import necessary libraries
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Set up environment variables (replace with your own keys)
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

# Define the function to read text from a PDF file
def read_pdf_text(filepath):
  """
  This function reads text content from a PDF file.

  Args:
      filepath (str): Path to the PDF file.

  Returns:
      str: Extracted text content from the PDF.
  """
  pdfreader = PdfReader(filepath)
  raw_text = ''
  for page in pdfreader.pages:
      content = page.extract_text()
      if content:
          raw_text += content
  return raw_text

# Specify the path to your PDF file
pdf_path = 'budget_speech.pdf'

# Read text from the PDF
raw_text = read_pdf_text(pdf_path)

# Text splitter configuration for efficient processing
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

# Split the raw text into smaller chunks
texts = text_splitter.split_text(raw_text)

# Download embeddings for text using OpenAI
embeddings = OpenAIEmbeddings()

# Create a document search index using FAISS for fast retrieval
document_search = FAISS.from_texts(texts, embeddings)

# Load the question answering chain with OpenAI for LLM interaction
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Define your questions
query1 = "Vision for Amrit Kaal"
query2 = "How much the agriculture target will be increased to and what the focus will be"

# Search for relevant documents in the index for each question
docs1 = document_search.similarity_search(query1)
docs2 = document_search.similarity_search(query2)

# Run the question answering chain with retrieved documents and questions
chain.run(input_documents=docs1, question=query1)
chain.run(input_documents=docs2, question=query2)

# (Optional) Load an online PDF document using OnlinePDFLoader
from langchain.document_loaders import OnlinePDFLoader

loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load()

# (Optional) Create a separate vector store index for online documents (requires chromadb)
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

# (Optional) Define a query for the online document (requires separate index)
query3 = "Explain me about Attention is all you need"
index.query(query3)
