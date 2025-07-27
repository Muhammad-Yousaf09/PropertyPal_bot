import streamlit as st
import os
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page config
st.set_page_config(
    page_title="Real Estate Assistant",
    page_icon="🏠"
)

class RealEstateChatbot:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.chat_model = None
        
    def test_api_key(self, openai_api_key):
        """Test if the API key is valid"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            st.error(f"Invalid API Key: {str(e)}")
            return False

    def initialize_components(self, openai_api_key):
        """Initialize all the RAG components"""
        try:
            # Test API key first
            if not self.test_api_key(openai_api_key):
                return False
                
            # Load and process documents
            if not os.path.exists("Property_data.csv"):
                st.error("Property_data.csv file not found!")
                return False
                
            with st.spinner("Loading and processing dataset..."):
                # Load CSV
                loader = CSVLoader(file_path="Property_data.csv")
                documents = loader.load()
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                )
                docs = text_splitter.split_documents(documents)
                
                # Create embeddings
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=openai_api_key
                )
                
                # Create vector store
                if os.path.exists("chroma_db"):
                    self.vectorstore = Chroma(
                        persist_directory="chroma_db",
                        embedding_function=embeddings
                    )
                else:
                    self.vectorstore = Chroma.from_documents(
                        docs,
                        embeddings,
                        persist_directory="chroma_db",
                    )
                
                # Create retriever
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # Create chat model
                self.chat_model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    openai_api_key=openai_api_key
                )
                
                # Create custom prompt template
                prompt_template = """
                You are a helpful real estate assistant. Use the following context to answer questions about properties.
                
                Context: {context}
                
                Question: {question}
                
                Please provide a detailed answer about the properties. Format your response as follows:
                - For each property, include: Location, Price, Area, Bedrooms, Baths, Date Added, Agency, Agent, and Page URL if available
                - Use clear formatting with bullet points or numbered lists
                - If no properties match the criteria, suggest similar alternatives
                - Be helpful and conversational
                
                Answer:"""
                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Create chain
                self.chain = RetrievalQA.from_chain_type(
                    llm=self.chat_model,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
            st.success("Chatbot initialized successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            return False

    def classify_query_intent(self, query):
        """Classify if the query is property-related or general conversation"""
        try:
            classification_prompt = f"""
            Analyze the following user query and determine if it's related to real estate/property search or general conversation.

            User Query: "{query}"

            Respond with ONLY one word:
            - "PROPERTY" if the query is about searching, finding, or asking about real estate, houses, apartments, properties, locations, prices, rooms, etc.
            - "GENERAL" if the query is a greeting, general conversation, personal question, or not related to real estate.

            Examples:
            - "hi how are you" → GENERAL
            - "hello" → GENERAL  
            - "what's your name" → GENERAL
            - "show me houses in karachi" → PROPERTY
            - "find 3 bedroom apartment" → PROPERTY
            - "properties under 50 lakh" → PROPERTY
            - "tell me about islamabad" → GENERAL (unless specifically asking about properties)
            - "thanks" → GENERAL

            Classification:"""

            response = self.chat_model.predict(classification_prompt)
            return response.strip().upper()
        except Exception as e:
            # If classification fails, default to PROPERTY to be safe
            return "PROPERTY"

    def get_general_response(self, query):
        """Get a general conversational response without RAG"""
        try:
            general_prompt = f"""
            You are a friendly and helpful Real Estate Assistant. The user is having a general conversation with you.
            
            User: {query}
            
            Respond in a conversational, helpful manner. Keep it brief and friendly. If appropriate, you can mention that you're here to help with property searches, but don't force it.
            
            Response:"""
            
            response = self.chat_model.predict(general_prompt)
            return response.strip(), []
        except Exception as e:
            return "Hello! I'm your Real Estate Assistant. How can I help you find the perfect property today?", []

    def get_property_response(self, query):
        """Get response using RAG for property-related queries"""
        try:
            result = self.chain({"query": query})
            return result['result'], result.get('source_documents', [])
        except Exception as e:
            return f"I'm sorry, I encountered an error while searching for properties: {str(e)}", []

    def get_response(self, query):
        """Main response method that routes based on query classification"""
        try:
            # Classify the query intent
            intent = self.classify_query_intent(query)
            
            # Route based on classification
            if intent == "GENERAL":
                return self.get_general_response(query)
            else:  # PROPERTY or any other classification defaults to property search
                return self.get_property_response(query)
                
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}", []

def main():
    # Header
    st.title("🏠 Real Estate Assistant")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password"
        )
        
        st.divider()
        
        st.header("💡 Example Queries")
        st.text("• Show me houses for sale in Karachi")
        st.text("• Find 3 bedroom houses in Islamabad") 
        st.text("• Properties under 50 lakh in Lahore")
        st.text("• Houses with 2+ bathrooms in DHA")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RealEstateChatbot()
        st.session_state.initialized = False
    
    # Initialize components if API key is provided
    if openai_api_key and not st.session_state.initialized:
        if st.session_state.chatbot.initialize_components(openai_api_key):
            st.session_state.initialized = True
    
    # Main chat interface
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to start chatting.")
        return
    
    if not st.session_state.initialized:
        st.error("Chatbot not initialized. Please check your API key and dataset.")
        return
    
    # Chat interface
    st.header("💬 Chat with Real Estate Assistant")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your Real Estate Assistant. How can I help you today?"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # User input
    user_query = st.chat_input("Ask about properties...")
    
    if user_query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get response from chatbot
        with st.spinner("Processing..."):
            response, source_docs = st.session_state.chatbot.get_response(user_query)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the display
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your Real Estate Assistant. How can I help you today?"
        }]
        st.rerun()

if __name__ == "__main__":
    main()