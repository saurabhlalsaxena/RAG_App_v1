__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langsmith import Client

client = Client()

# Function to split PDF into chunks for summarisation
def get_tokenSplit(pages):
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
  chunk = {}
  total_tokens = 0
  string = ""
  i = 0
  for page in pages:
    tokens_page = len(enc.encode(page.page_content))
    total_tokens += tokens_page
    if total_tokens < 15000:
      string = string + " " + page.page_content
      chunk[i] = string
    else:
      #print(chunk.keys(), ":", total_tokens)
      string = page.page_content
      total_tokens = 0
      i += 1
  return chunk

#Langchain function to call model completion with specified prompt
def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

#Function to get Section(Token Splits) Summaries
def get_sectionSummaries(chunks):
  from langchain_openai import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate

  chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k")

  template_string = """You are the editor of Harvard Business Review.\
    Read the section of the document added below and summarise it in 500 words or less.
    The summary should contain only the topics discussed in the document and the main insights. Make sure not to exceed 150 words.
    text: ```{text}```
    """

  sectionSummaries = {}
  for chunk in chunks:
    print (chunk)
    from langchain.prompts import ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(template_string)
    book_section = prompt_template.format_messages(text=chunks[chunk])
    section_summary = chat.invoke(book_section)
    sectionSummaries[chunk] = section_summary

  stringSectionSummaries = ""
  for sectionSummary in sectionSummaries:
    stringSectionSummaries = stringSectionSummaries + " " + sectionSummaries[sectionSummary].content

  return sectionSummaries, stringSectionSummaries

#Function to get Book Summary from Section Summaries
def get_bookSummaries(stringSectionSummaries):
  from langchain_openai import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate

  chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k")

  template_string2 = """You are the editor of Harvard Business Review.\
  Read the below section summaries of a document and summarise the complete document based on the section summaries.
  The summary should contain only the topics discussed in the document and the main insights. The response should not refer to the section summaries
  and only present a complete summary of the document. Mention the word summary.
  text: ```{text}```
  """

  prompt_template2 = ChatPromptTemplate.from_template(template_string2)
  book_sectionSummaries = prompt_template2.format_messages(
                    text=stringSectionSummaries)
  book_summary = chat.invoke(book_sectionSummaries)
  return book_summary

# Function to get Splits for VectorDB
def get_splitsForVectorDB(pages):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1500,
      chunk_overlap = 150
  )
  splits = text_splitter.split_documents(pages)
  return splits

#Function to create of vector store
def create_VectorStore(splits):
  from langchain_openai import OpenAIEmbeddings
  from langchain_community.vectorstores import Chroma

  embeddings = OpenAIEmbeddings()

  persist_directory = 'chroma/'
  vectordb = Chroma.from_documents(
      documents=splits,
      embedding=embeddings,
      persist_directory=persist_directory
  )

  vectordb=Chroma(
      persist_directory=persist_directory,
      embedding_function = embeddings
  )
  return vectordb

#Function to answer questions
def question_answer(vectordb, question):
  from langchain.prompts import PromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain.chains import RetrievalQA

  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  # Build prompt
  template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question is conversational, then answer in a conversational tone. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
  {context}
  Question: {question}
  Helpful Answer:"""
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

  # Run chain
  qa_chain = RetrievalQA.from_chain_type(
      llm,
      retriever=vectordb.as_retriever(),
      return_source_documents=True,
      chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
  )
  result = qa_chain.invoke({"query": question})
  return result


#Function to answer questions with memory
def question_answerWithMemory (vectordb):
  from langchain.prompts import PromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain.memory import ConversationBufferMemory
  from langchain.chains import ConversationalRetrievalChain

  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  memory = ConversationBufferMemory(
      memory_key="chat_history",
      return_messages=True
  )

  # Run chain
  retriever=vectordb.as_retriever()
  qa = ConversationalRetrievalChain.from_llm(
      llm,
      retriever=retriever,
      memory=memory
  )
  return qa


#Function to load PDF
def loadpdf(uploaded_file):
  file_name = uploaded_file.name
  temp_file = f"./{file_name}"
  with open(temp_file, "wb") as file:
      file.write(uploaded_file.getvalue())
  return temp_file

def main():
  st.set_page_config(page_title = "Chat with your documents")
  st.title('Chat with your documents!')
  text = st.chat_input("Type your message here...")

  st.sidebar.header("Settings")

  uploaded_file = st.sidebar.file_uploader('Choose your PDF file', type = "pdf")

  # session state
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = [
          AIMessage(content="Welcome! 🌟 Feel free to upload a document, and I'll assist you with any questions or insights you need. Whether it's summarizing content, answering queries, or discussing key points, I'm here to help enhance your understanding. Simply upload your document, and let's get started on our insightful journey together!"),
      ]

  if uploaded_file:
    if 'vs' not in st.session_state:
      temp_file = loadpdf(uploaded_file)
      loader = PyPDFLoader(temp_file)
      pages = loader.load()

  #Managing new document uploads
  if uploaded_file is None:
    if 'vs' in st.session_state:
      del st.session_state.vs

  if uploaded_file and 'vs' not in st.session_state:
    #Call functions for summarisation
    with st.sidebar:
      with st.spinner('Generating Summary...'):
        chunks = get_tokenSplit(pages)
        sectionSummaries, stringSectionSummaries = get_sectionSummaries(chunks)
        book_summary = get_bookSummaries(stringSectionSummaries)


    #print(book_summary.content)
    summary = book_summary.content
    st.session_state.chat_history.append(AIMessage(content=summary))

    #Call functions for creating VectorDB
    splits = get_splitsForVectorDB(pages)
    vectordb = create_VectorStore(splits)
    vectordb.persist()

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    st.session_state.vs = vectordb
    st.sidebar.success('Uploaded, chunked and embedded successfully.')

  #if submitted and 'vs' in st.session_state:
  if text is not None and text !="" and 'vs' in st.session_state:
      question= text
      vectordb = st.session_state.vs
      if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = question_answerWithMemory(vectordb)
      result = st.session_state.qa_chain.invoke({"question": question})
      st.session_state.chat_history.append(HumanMessage(content=question))
      st.session_state.chat_history.append(AIMessage(content=result['answer']))

  # conversation
  for message in st.session_state.chat_history:
      if isinstance(message, AIMessage):
          with st.chat_message("Bot"):
              st.write(message.content)
      elif isinstance(message, HumanMessage):
          with st.chat_message("User"):
              st.write(message.content)


if __name__ == "__main__":
    main()
