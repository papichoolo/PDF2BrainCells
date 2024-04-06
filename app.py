import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.callbacks import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os
load_dotenv()
MYKEY=str(os.getenv('OPENAI'))
# Set page title
st.title('PDF2BrainCells 📑➡️🧠')

# Upload PDF file
uploaded_file = st.sidebar.file_uploader('Upload a PDF file', type=['pdf'])
user_api_key = st.sidebar.text_input('Enter your OpenAI API key (optional)',type='password')
if uploaded_file:
    # Perform text processing
    temp_file_path = 'temp.pdf'
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_file_path)
    pages= loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=user_api_key if user_api_key else MYKEY))
    retriever = faiss_index.as_retriever()
    template = """Answer the question based for an examination point of view only on the following context in Markdown format. :
        {context}

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    streamingcall=StreamingStdOutCallbackHandler()
    model = ChatOpenAI(openai_api_key=user_api_key if user_api_key else MYKEY,streaming=True,callbacks=[streamingcall],callback_manager=None)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
    # Select operation
    option = st.selectbox('Select an operation', ['Generate Questions','Conversation on Text','Content Structure','Summarization'])

    if option == 'Summarization':
        # Perform summarization
        summary=chain.invoke("Provide an executive summary of the document")
        st.write(summary)
    elif option == 'Content Structure':
        # Perform structure extraction
        structure=chain.invoke("Provide a structure of contents for this document")
        print(structure)
        st.write(structure)
    elif option == 'Generate Questions':
        # Generate questions based on content
        questions = chain.invoke("Generate questions based on the content of the document")
        st.header('Generated Questions:')
        st.markdown(questions)
    elif option == 'Conversation on Text':
        def generate_response(input_text):
            # Create a placeholder for the response
            response_placeholder = st.empty()

            # Initialize an empty string to store the response
            response = ""

            # Use a context manager to stream the response
            with st.spinner('Generating response...'):
                # Call your existing chain to get the response stream
                stream = chain.stream(input_text + " \n Give a detailed answer in an examination point of view")
                # Stream the response chunk by chunk
                for chunk in stream:
                    # Append the current chunk to the response
                    response += chunk

                    # Update the placeholder with the current response
                    response_placeholder.markdown(response)
    # Add a horizontal line after the response is complete
            #st.markdown("---")

        with st.form('my_form'):
            text = st.text_area('Enter text:', 'What is this Document About?')
            submitted = st.form_submit_button('Submit')
            if not user_api_key.startswith('sk-'):
                st.toast('It is preferrable to use your own OpenAI API key for continued working of the app', icon='❓')
            if submitted:
                generate_response(text)
            # Add a button to the sidebar
            