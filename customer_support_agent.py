"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    LOVABLE FRONTEND INTEGRATION                             ║
╚════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS FOR COPILOT:
When the section below is updated with new Lovable frontend code, automatically 
update the backend API to match the request/response format shown.

╔════════════════════════════════════════════════════════════════════════════╗
║                    CURRENT FRONTEND API CALL                                ║
║                    (Paste your Lovable code here)                           ║
╚════════════════════════════════════════════════════════════════════════════╝

/**
 * API client for the Python RAG backend
 * Replace API_BASE_URL with your actual Python backend endpoint
 */

const API_BASE_URL = import.meta.env.VITE_SUPPORT_API_URL || 'http://localhost:8000';

export interface ChatMessage {
  question: string;
}

export interface ChatResponse {
  answer: string;
  needs_human_support: boolean;
  context?: Array<{
    content: string;
    category: string;
  }>;
}

export const sendSupportMessage = async (message: string): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: message }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error calling support API:', error);
    // Return a fallback response if API is unavailable
    return {
      answer: "I'm currently unable to process your question. Please try again later or contact support via WhatsApp.",
      needs_human_support: true,
    };
  }
};

export const WHATSAPP_SUPPORT_LINK = import.meta.env.VITE_WHATSAPP_LINK || 'https://wa.me/1234567890';

Last Updated: 2025-10-16
Frontend Repo: https://github.com/Tsarri/kntrkt-nomad-assist-81125

╔════════════════════════════════════════════════════════════════════════════╗
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List
from supabase import create_client, Client
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Set up Mistral API key automatically
os.environ["MISTRAL_API_KEY"] = "0OYFp5b31qOfBEYVppnkUlDmn4uuLen4"

# Your business WhatsApp link
WHATSAPP_SUPPORT_LINK = "https://wa.me/1234567890"  # UPDATE THIS

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connected")
else:
    print("⚠️  Supabase not configured - running without persistence")

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

def load_spreadsheet_documents():
    """Load documents from Excel or CSV files"""
    documents = []
    
    if not os.path.exists(support_docs_path):
        print(f"Warning: {support_docs_path} does not exist")
        return documents
    
    for filename in os.listdir(support_docs_path):
        filepath = os.path.join(support_docs_path, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        try:
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
                # Format 1: Question/Answer style
                if 'Question' in df.columns and 'Answer' in df.columns:
                    content = f"Question: {row['Question']}\n\nAnswer: {row['Answer']}"
                    category = row.get('Category', 'General')
                
                # Format 2: Topic/Content style
                elif 'Topic' in df.columns and 'Content' in df.columns:
                    content = f"Topic: {row['Topic']}\n\n{row['Content']}"
                    category = row.get('Category', 'General')
                
                # Format 3: Generic
                else:
                    columns = df.columns.tolist()
                    if len(columns) >= 2:
                        content = f"{columns[0]}: {row[columns[0]]}\n\n{columns[1]}: {row[columns[1]]}"
                        category = row.get(columns[2], 'General') if len(columns) > 2 else 'General'
                    else:
                        continue
                
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

# Load documents
print("Loading support documentation...")
docs = load_spreadsheet_documents()

if len(docs) == 0:
    print("\nWARNING: No documents were loaded!")
    print("Add .xlsx or .csv files to the SupportDocuments folder")
else:
    print(f"Loaded {len(docs)} support entries")
    vector_store.add_documents(documents=docs)
    print("Documentation indexed successfully\n")

# Create the prompt template
support_prompt_template = """You are a helpful customer support agent for a digital contract platform that uses Stripe for payments.

User Question: {question}

Relevant Documentation:
{context}

Instructions:
- Answer based on the documentation
- Be friendly, concise, and helpful
- Use simple language
- If you can't fully answer or the issue is complex, say so honestly
- For complex issues, mention contacting human support

Provide a clear, helpful response:"""

support_prompt = PromptTemplate.from_template(support_prompt_template)

def get_ai_response(question: str) -> dict:
    """Generate AI response for a customer question"""
    try:
        # Retrieve relevant documents
        if len(docs) > 0:
            retrieved_docs = vector_store.similarity_search(question, k=3)
        else:
            return {
                "answer": "I don't have access to support documentation right now. Please contact our support team for assistance.",
                "needs_human_support": True,
                "context": []
            }
        
        # Format context
        docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate response
        messages = support_prompt.invoke({
            "question": question,
            "context": docs_content
        })
        response = llm.invoke(messages)
        answer = response.content
        
        # Check if human support is needed
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
            "needs_human_support": needs_human,
            "context": [
                {
                    "content": doc.page_content,
                    "category": doc.metadata.get("category", "General")
                }
                for doc in retrieved_docs
            ]
        }
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            "answer": "I'm experiencing technical difficulties. Please contact our support team.",
            "needs_human_support": True,
            "context": []
        }

# ============================================================================
#                              API ENDPOINTS
# ============================================================================

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint - receives customer questions and returns AI responses
    Now creates/updates tickets in Supabase for admin dashboard
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' field in request",
                "needs_human_support": True
            }), 400
        
        question = data['question']
        ticket_id = data.get('ticket_id', None)
        customer_info = data.get('customer_info', {})
        
        # Get AI response
        response = get_ai_response(question)
        
        # Store in Supabase if configured
        if supabase:
            try:
                # Create new ticket if no ticket_id provided
                if not ticket_id:
                    ticket_data = {
                        "ticket_id": str(uuid.uuid4()),
                        "customer_name": customer_info.get('name', 'Anonymous'),
                        "customer_email": customer_info.get('email', ''),
                        "customer_location": customer_info.get('location', ''),
                        "customer_user_type": customer_info.get('user_type', 'Customer'),
                        "inquiry_text": question,
                        "inquiry_type": customer_info.get('inquiry_type', 'general'),
                        "priority": 'high' if response.get('needs_human_support') else 'medium',
                        "status": "ai_responded" if not response.get('needs_human_support') else "new",
                        "ai_confidence_score": 0.85 if not response.get('needs_human_support') else 0.5,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    supabase.table('tickets').insert(ticket_data).execute()
                    ticket_id = ticket_data['ticket_id']
                    
                    # Store customer message
                    supabase.table('messages').insert({
                        "message_id": str(uuid.uuid4()),
                        "ticket_id": ticket_id,
                        "sender_type": "customer",
                        "message_text": question,
                        "created_at": datetime.utcnow().isoformat()
                    }).execute()
                
                # Store AI response message
                supabase.table('messages').insert({
                    "message_id": str(uuid.uuid4()),
                    "ticket_id": ticket_id,
                    "sender_type": "ai",
                    "message_text": response['answer'],
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                
            except Exception as e:
                print(f"Error saving to Supabase: {e}")
        
        # Add ticket_id to response
        response['ticket_id'] = ticket_id
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "needs_human_support": True
        }), 500

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """
    Get all tickets for admin dashboard
    Supports filtering by status
    """
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 503
    
    try:
        status_filter = request.args.get('status', None)
        
        query = supabase.table('tickets').select('*').order('created_at', desc=True)
        
        if status_filter:
            query = query.eq('status', status_filter)
        
        result = query.execute()
        
        return jsonify({
            "tickets": result.data,
            "total": len(result.data)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tickets/<ticket_id>', methods=['GET'])
def get_ticket_detail(ticket_id):
    """
    Get specific ticket details with conversation history
    """
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 503
    
    try:
        # Get ticket
        ticket_result = supabase.table('tickets').select('*').eq('ticket_id', ticket_id).execute()
        
        if not ticket_result.data:
            return jsonify({"error": "Ticket not found"}), 404
        
        # Get messages
        messages_result = supabase.table('messages').select('*').eq('ticket_id', ticket_id).order('created_at', desc=False).execute()
        
        ticket = ticket_result.data[0]
        ticket['messages'] = messages_result.data
        
        return jsonify(ticket), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tickets/<ticket_id>', methods=['PATCH'])
def update_ticket(ticket_id):
    """
    Update ticket (e.g., mark as resolved)
    """
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No update data provided"}), 400
        
        # Add updated_at timestamp
        data['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase.table('tickets').update(data).eq('ticket_id', ticket_id).execute()
        
        if not result.data:
            return jsonify({"error": "Ticket not found"}), 404
        
        return jsonify({
            "success": True,
            "ticket": result.data[0]
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tickets/summary', methods=['GET'])
def get_tickets_summary():
    """
    Get summary counts for admin dashboard
    """
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 503
    
    try:
        # Get counts by status
        all_tickets = supabase.table('tickets').select('status').execute()
        
        resolved_count = sum(1 for t in all_tickets.data if t['status'] == 'resolved')
        unresolved_count = len(all_tickets.data) - resolved_count
        
        return jsonify({
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "total": len(all_tickets.data)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "documentsLoaded": len(docs),
        "supabaseConnected": supabase is not None
    }), 200

# ============================================================================
#                           APPLICATION STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Customer Support RAG API Server")
    print("="*60)
    print(f"\nAPI running on: http://localhost:8000")
    print(f"Chat endpoint: http://localhost:8000/chat")
    print(f"Health check: http://localhost:8000/api/health")
    print(f"\nSupabase: {'✅ Connected' if supabase else '⚠️  Not configured'}")
    print("\nReady to receive requests from Lovable frontend!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
