import os
import pandas as pd
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Set up Mistral API key automatically
os.environ["MISTRAL_API_KEY"] = "0OYFp5b31qOfBEYVppnkUlDmn4uuLen4"

# Your business WhatsApp link
WHATSAPP_SUPPORT_LINK = "https://wa.me/1234567890"  # UPDATE THIS with your actual WhatsApp number

# Initialize AI components
print("Initializing customer support agent...")
llm = ChatMistralAI(
    model="mistral-large-latest",
    timeout=120.0
)
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embeddings)

# Path to support documents
support_docs_path = os.path.join(os.path.dirname(__file__), "SupportDocuments")

# Load support documents from spreadsheet
print("Loading support documentation from spreadsheet...")

def load_spreadsheet_documents():
    """Load documents from Excel or CSV files in the SupportDocuments folder"""
    documents = []
    
    # Look for spreadsheet files in the SupportDocuments folder
    for filename in os.listdir(support_docs_path):
        filepath = os.path.join(support_docs_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(filepath):
            continue
        
        try:
            # Load Excel or CSV file
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(filepath)
                print(f"Loaded Excel file: {filename}")
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                print(f"Loaded CSV file: {filename}")
            else:
                continue
            
            # Process each row as a document
            for index, row in df.iterrows():
                # Try different column name combinations
                # Format 1: Question/Answer style
                if 'Question' in df.columns and 'Answer' in df.columns:
                    content = f"Question: {row['Question']}\n\nAnswer: {row['Answer']}"
                    category = row.get('Category', 'General')
                
                # Format 2: Topic/Content style
                elif 'Topic' in df.columns and 'Content' in df.columns:
                    content = f"Topic: {row['Topic']}\n\n{row['Content']}"
                    category = row.get('Category', 'General')
                
                # Format 3: Generic - use first two columns
                else:
                    columns = df.columns.tolist()
                    if len(columns) >= 2:
                        content = f"{columns[0]}: {row[columns[0]]}\n\n{columns[1]}: {row[columns[1]]}"
                        category = row.get(columns[2], 'General') if len(columns) > 2 else 'General'
                    else:
                        continue
                
                # Create a document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "category": category,
                        "row": index
                    }
                )
                documents.append(doc)
        
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return documents

# Load the documents
docs = load_spreadsheet_documents()

if len(docs) == 0:
    print("\nWARNING: No documents were loaded!")
    print("Please make sure you have a .xlsx or .csv file in the SupportDocuments folder")
    print("with columns like: Question, Answer, Category")
    print("or: Topic, Content, Category\n")
else:
    print(f"Loaded {len(docs)} support entries from spreadsheet\n")

# Store in vector database
print("Indexing documents...")
if docs:
    vector_store.add_documents(documents=docs)
    print("Support documentation indexed successfully\n")

# Create the prompt template for the support agent
support_prompt_template = """You are a helpful customer support agent for a digital contract platform that uses Stripe for payments.

User Question: {question}

Relevant Documentation:
{context}

Instructions:
- Answer the user's question based on the provided documentation
- Be friendly, concise, and helpful
- Use simple language that's easy to understand
- If the documentation doesn't contain enough information to fully answer the question, or if the issue seems complex, say so honestly
- For complex issues you can't resolve, mention that the user should contact a human support agent

Provide a clear, helpful response:"""

support_prompt = PromptTemplate.from_template(support_prompt_template)

# Define the state for the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    needs_human_support: bool

def retrieve(state: State):
    """Retrieve relevant documentation based on the user's question"""
    question = state["question"]
    
    # Search for relevant documents
    if len(docs) > 0:
        retrieved_docs = vector_store.similarity_search(question, k=3)
    else:
        retrieved_docs = []
    
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate a response based on the retrieved documentation"""
    try:
        # Check if we have any context
        if not state["context"]:
            return {
                "answer": "I don't have enough information in my knowledge base to answer that question. Let me connect you with a human support agent who can help.",
                "needs_human_support": True
            }
        
        # Format the context
        docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
        
        # Generate the response
        messages = support_prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })
        response = llm.invoke(messages)
        answer = response.content
        
        # Check if the answer indicates uncertainty or complexity
        uncertainty_indicators = [
            "don't have enough information",
            "not sure",
            "unclear",
            "complex issue",
            "contact support",
            "reach out to",
            "speak with"
        ]
        
        needs_human = any(indicator in answer.lower() for indicator in uncertainty_indicators)
        
        return {
            "answer": answer,
            "needs_human_support": needs_human
        }
    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            "answer": "I'm having trouble processing your question right now. Please contact our support team for assistance.",
            "needs_human_support": True
        }

# Build the application graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def chat_interface():
    """Main chat interface for the support agent"""
    print("="*60)
    print("Customer Support Agent - Digital Contract Platform")
    print("="*60)
    print("\nHello! I'm here to help you with any questions about our")
    print("digital contract platform and payment processing.")
    print("\nType 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_question = input("You: ").strip()
        
        # Check for exit commands
        if user_question.lower() in ['exit', 'quit', 'bye']:
            print("\nAgent: Thank you for contacting support. Have a great day!")
            break
        
        # Skip empty input
        if not user_question:
            continue
        
        # Process the question
        print("\nAgent: Let me check that for you...\n")
        
        try:
            response = graph.invoke({"question": user_question})
            
            # Display the answer
            print(f"Agent: {response['answer']}\n")
            
            # If human support is needed, provide WhatsApp link
            if response.get('needs_human_support', False):
                print(f"For further assistance, you can reach our support team via WhatsApp:")
                print(f"{WHATSAPP_SUPPORT_LINK}\n")
        
        except Exception as e:
            print(f"Agent: I apologize, but I'm experiencing technical difficulties.")
            print(f"Please contact our support team via WhatsApp for immediate help:")
            print(f"{WHATSAPP_SUPPORT_LINK}\n")

# Main execution
if __name__ == "__main__":
    chat_interface()