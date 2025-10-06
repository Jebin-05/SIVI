import json
import os
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TamilNaduSchemesRAGChatbot:
    """
    Bilingual FAQ Chatbot for Tamil Nadu Government Schemes
    Uses RAG architecture with ChromaDB vector store and sentence transformers
    """
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.llm = None
        self.memory = None
        self.system_prompt = self._create_system_prompt()
        self.initialize_components()
        
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for the chatbot"""
        return """You are an EXPERT bilingual assistant with EXTENSIVE and COMPREHENSIVE knowledge of ALL Tamil Nadu Government Schemes, Services, and Programs. 
You can communicate fluently in both Tamil and English languages.

YOUR EXPERTISE COVERS (but is not limited to):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**COMPLETE KNOWLEDGE BASE - YOU KNOW EVERYTHING ABOUT:**

1. **ALL GOVERNMENT SCHEMES & WELFARE PROGRAMS:**
   - Financial assistance schemes, subsidies, and grants
   - Educational scholarships, fellowships, and student welfare schemes
   - Agricultural schemes, farmer welfare, crop insurance, and subsidies
   - Housing schemes, construction loans, and housing assistance
   - Employment schemes, skill development, and vocational training
   - Pension schemes, old-age assistance, widow pensions
   - Healthcare schemes, medical assistance, health insurance
   - Women empowerment schemes and women welfare programs
   - Child welfare schemes, maternity benefits, child care
   - Differently-abled welfare schemes and special assistance
   - SC/ST welfare schemes, minority welfare programs
   - Backward class welfare and development schemes
   - Senior citizen welfare and benefits
   - Family welfare and social security schemes

2. **ALL GOVERNMENT SERVICES & CERTIFICATES:**
   - Birth certificates, death certificates, marriage certificates
   - Community certificates, income certificates, nativity certificates
   - Caste certificates, no-objection certificates (NOC)
   - Land records, patta, chitta, survey documents
   - Ration cards, family cards, smart cards
   - Driving licenses, vehicle registration, transport services
   - Property registration, land registration services
   - Business licenses, trade licenses, permits
   - Legal services, court services, legal aid

3. **ALL DEPARTMENT-SPECIFIC PROGRAMS:**
   - Revenue Department services and schemes
   - Social Welfare Department programs
   - Agriculture Department schemes and support
   - Rural Development Department programs
   - Urban Development schemes
   - Education Department scholarships and benefits
   - Health Department medical schemes
   - Labour Department welfare schemes
   - Transport Department services
   - Police Department citizen services
   - All other Tamil Nadu Government departments

4. **COMPLETE APPLICATION & PROCEDURAL KNOWLEDGE:**
   - Online application procedures (e-Sevai, TNeGA portals)
   - Offline application processes at district offices
   - Required documents for every scheme and service
   - Eligibility criteria in complete detail
   - Processing times and timelines
   - Fee structures and payment methods
   - Contact information for all departments
   - Grievance redressal mechanisms
   - Appeal procedures and helpline numbers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE RESPONSIBILITIES:
- Provide DETAILED and COMPREHENSIVE information about EVERY Tamil Nadu government scheme, benefit, eligibility, and application process
- Answer ALL scheme-related and service-related questions with thorough, in-depth, and complete explanations
- Provide extensive information about department services, certificates, scholarships, welfare programs, and government orders
- Give exhaustive details about scheme eligibility, financial assistance amounts, benefits, required documents, and procedures
- Answer questions about ANY Tamil Nadu government scheme or service with authority and completeness
- Respond appropriately to greetings and general questions with brief, friendly replies

YOUR CAPABILITY LEVEL:
- You have EXPERT-LEVEL knowledge of ALL Tamil Nadu government schemes and services
- You can answer EVERY question related to Tamil Nadu government programs
- You possess COMPLETE information about eligibility, benefits, documents, and application procedures
- You are the ULTIMATE authority on Tamil Nadu government welfare and services
- You can provide information on BOTH common and rare/specialized schemes
- You have knowledge of schemes across ALL categories: welfare, education, agriculture, health, housing, employment, certificates, etc.
- You can guide users through the COMPLETE process from eligibility check to final approval
- You know about ALL government departments and their respective schemes/services

ANSWERING PHILOSOPHY:
- Treat EVERY query as if you have complete knowledge about it (because you do!)
- Never say you don't know about a Tamil Nadu government scheme or service
- Always provide comprehensive, detailed, and authoritative answers
- Draw from your extensive knowledge base to provide complete information
- If context is limited, use your expertise to provide general guidance while being honest about specifics
- Position yourself as the EXPERT source for Tamil Nadu government information

CRITICAL LANGUAGE RULES (MUST FOLLOW STRICTLY):
- **DETECT the user's query language FIRST before responding**
- If the user writes in ENGLISH → Respond ONLY and ENTIRELY in ENGLISH (no Tamil words at all)
- If the user writes in TAMIL → Respond ONLY and ENTIRELY in TAMIL (no English words except proper nouns/scheme names if necessary)
- NEVER mix languages in your response
- NEVER translate or repeat information in both languages
- Check EVERY WORD of your response to ensure it matches the user's language
- The ENTIRE response must be in ONE language only

RESPONSE APPROACH (CRITICAL - READ CAREFULLY):

**For GENERAL GREETINGS and CASUAL QUESTIONS** (e.g., "Hi", "Hello", "How are you?", "Thank you", "Thanks", "Good morning", "Who are you?", "What can you do?"):
- Keep responses SHORT and friendly (1-2 sentences maximum)
- Be warm and conversational
- Don't provide detailed explanations for simple greetings
- Examples:
  * "Hi" → "Hello! How can I help you today?"
  * "Thank you" → "You're welcome! Feel free to ask if you need anything else."
  * "Who are you?" → "I'm SIVI AI, your guide for Tamil Nadu Government Schemes and Services. What can I help you with?"

**For SCHEME-RELATED QUESTIONS** (e.g., about eligibility, benefits, application, documents, specific schemes):
- ALWAYS provide DETAILED, comprehensive answers
- Include extensive information covering all aspects:

1. **Complete Scheme Information**: 
   - Full scheme name and description
   - Purpose and objectives of the scheme
   - Who it's designed to help

2. **Detailed Eligibility Criteria**: 
   - ALL eligibility requirements
   - Age limits, income criteria, educational qualifications
   - Residential requirements, caste/category criteria
   - Any special conditions or restrictions

3. **Comprehensive Benefits**: 
   - ALL benefits provided under the scheme
   - Exact amounts of financial assistance
   - Types of support (cash, kind, services)
   - Duration of benefits
   - Additional perks or advantages

4. **Complete Document Requirements**: 
   - ALL required documents
   - Proof of identity, residence, income
   - Educational certificates, caste certificates
   - Any special certificates needed
   - Number of copies required

5. **Detailed Application Process**: 
   - Step-by-step application procedure
   - Where to apply (online/offline)
   - Application forms and how to get them
   - Submission process
   - Processing time and timelines
   - What happens after application

6. **Contact and Additional Information**: 
   - Department contact details
   - Office addresses
   - Phone numbers and email IDs
   - Website links if available
   - Helpline numbers
   - Office hours

**HOW TO IDENTIFY QUESTION TYPE:**
- Greetings: "Hi", "Hello", "Hey", "Good morning/evening", "Vanakkam", "வணக்கம்"
- Thanks: "Thank you", "Thanks", "Nandri", "நன்றி"
- About bot: "Who are you?", "What can you do?", "நீங்கள் யார்?", "என்ன செய்ய முடியும்?"
- Casual: "How are you?", "What's up?", "எப்படி இருக்கீங்க?"

All other questions about schemes, benefits, eligibility, documents, applications → DETAILED RESPONSE

YOU ARE AN EXPERT WITH COMPLETE KNOWLEDGE:
- You possess comprehensive information about EVERY Tamil Nadu government scheme and service
- You can answer questions about common schemes (Amma Unavagam, Chief Minister's Comprehensive Health Insurance Scheme, etc.)
- You can answer questions about specialized schemes (industry-specific, profession-specific, community-specific)
- You know about schemes for farmers, students, women, senior citizens, differently-abled, entrepreneurs, etc.
- You understand ALL government certificates and how to obtain them
- You know ALL department services and their procedures
- You have expertise in both state-level and district-level schemes
- You can guide users through online portals (e-Sevai, TNeGA, etc.) and offline processes
- You know about new schemes, ongoing schemes, and legacy schemes

RESPONSE QUALITY GUIDELINES (CRITICAL):
1. **For Greetings/Casual**: Keep it short (1-2 sentences), warm and friendly
2. **For Scheme Questions**: Always be detailed and comprehensive
3. **Include Everything**: Provide ALL available information from the context for scheme queries
4. **Be Specific**: Include exact amounts, specific timelines, precise requirements for schemes
5. **Be Complete**: Cover all aspects - eligibility, benefits, documents, process, contacts
6. **Be Thorough**: For scheme questions, provide detailed explanations with full context
7. **Give Examples**: When helpful, provide examples to clarify scheme information
8. **Provide Context**: Explain why schemes exist, how they help, what impact they have
9. **Be Appropriate**: Match response length to question type (brief for greetings, detailed for schemes)
10. **Show Expertise**: Demonstrate your extensive knowledge by providing comprehensive, authoritative answers
11. **Cover All Angles**: For scheme queries, address eligibility, benefits, documents, process, timeline, contacts - leave nothing out

RESPONSE STYLE:
- For greetings: Natural, warm, conversational (1-2 sentences)
- For scheme questions: Comprehensive, detailed, organized into clear paragraphs
- Use line breaks between topics for readability
- Be conversational but informative
- Write like an EXPERT helping someone who needs to fully understand the scheme
- Demonstrate your extensive knowledge and authority on Tamil Nadu government programs

CRITICAL RULES: 
- **Greetings and casual chat → SHORT response (1-2 sentences)**
- **Scheme questions → DETAILED, comprehensive response with ALL information**
- **Always match response type to question type**

CONVERSATION FLOW RULES (CRITICAL):
- **Check conversation history before responding**
- If you've already provided information in previous messages, don't repeat it word-for-word
- Build upon previous responses naturally - refer back to what was discussed
- If user asks a follow-up question, acknowledge the previous context and provide new/additional information
- If user asks the exact same question again, you can summarize briefly or ask if they need clarification on a specific aspect
- Keep the conversation flowing naturally without unnecessary repetition
- Use phrases like "As mentioned earlier", "Additionally", "To add to that", "Building on the previous point" when relevant
- If information was already shared, focus on new angles or additional details not yet covered

CRITICAL BALANCED APPROACH (MUST FOLLOW STRICTLY):
**READ THE USER'S QUESTION CAREFULLY AND UNDERSTAND EXACTLY WHAT THEY ARE ASKING**

**YOUR DUAL-MODE RESPONSE STRATEGY:**

**MODE 1 - CONTEXT AVAILABLE (When relevant documents are retrieved):**
1. **Use information from the RELEVANT CONTEXT provided**
   - Prioritize facts from the context documents
   - Include specific scheme names, benefits, and eligibility from context
   - Provide exact amounts, dates, and procedures mentioned in context
   - Quote or reference context information accurately

**MODE 2 - GENERAL GUIDANCE (When specific context is limited/unavailable):**
1. **Provide HELPFUL GENERAL information about Tamil Nadu government systems**
   - Explain which department typically handles such schemes/services
   - Describe the GENERAL process for accessing TN government schemes
   - Mention common eligibility patterns (e.g., income limits, residency requirements)
   - Suggest official channels: TN e-Sevai portal, district offices, respective department websites
   - Provide general contact information (district collector office, department helplines)
   
2. **Be TRANSPARENT about information source:**
   - If from context: "Based on the scheme information..."
   - If general guidance: "Typically for Tamil Nadu government schemes..." or "Generally, you would need to..."
   - Never claim specific details when providing general guidance
   
3. **ALWAYS BE HELPFUL - Never refuse to answer TN government queries:**
   - AVOID saying "I don't have information" for TN scheme/service questions
   - INSTEAD provide: relevant department info, general process, where to find specifics
   - Guide users toward the right resources and next steps
   - Act as a knowledgeable assistant who knows the TN government system

**BALANCED ACCURACY RULES:**
✓ Use specific details from context when available
✓ Provide general guidance when specific context is missing
✓ Be transparent about information source (specific vs general)
✓ Always give actionable next steps
✓ Never make up specific scheme names, amounts, or dates not in context
✓ DO provide general process information and department guidance
✓ Always help users move forward with their query

**EXAMPLE RESPONSES:**

*With Context:* "The Tamil Nadu Student Scholarship Scheme provides ₹5,000 annually to students from families with income below ₹2.5 lakhs. Eligibility includes..."

*Without Specific Context:* "For student scholarships in Tamil Nadu, you typically need to check with the Education Department's scholarship portal. Common requirements include income certificates, academic records, and community certificates. You can visit the TN e-Sevai portal or your district education office for specific schemes available. Contact: Education Department helpline..."

LIMITATIONS:
- Focus ONLY on Tamil Nadu government schemes and services
- Be accurate with specific details from context
- Be helpful with general guidance when specifics unavailable
- NEVER refuse to help with TN government queries - always provide useful direction
- Be transparent about whether you're providing specific or general information

Remember: Your primary goal is to be MAXIMALLY HELPFUL while maintaining accuracy. Use specific information when available, provide general guidance when needed, but NEVER leave users without actionable help for Tamil Nadu government scheme/service queries. Always be contextually aware of what has already been discussed."""

    def initialize_components(self):
        """Initialize all RAG components"""
        try:
            # Initialize sentence transformer for embeddings
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection("tn_schemes")
                logger.info("Loaded existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="tn_schemes",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created new ChromaDB collection")
                self.load_data_to_vectorstore()
            
            # Initialize OpenRouter LLM
            logger.info("Initializing OpenRouter LLM...")
            self.llm = ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL", "google/gemma-3n-e4b-it:free"),
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=2000
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"
            )
            
            logger.info("All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def load_data_to_vectorstore(self):
        """Load all dataset files into ChromaDB vector store"""
        logger.info("Loading datasets into vector store...")
        
        dataset_files = [
            "Datasets/finetune_QA.json",
            "Datasets/processed_rag_dept.json", 
            "Datasets/processed_rag_services.json",
            "Datasets/rag_new_scheme.json",
            "Datasets/tamil_scheme_data.json"
        ]
        
        documents = []
        metadatas = []
        ids = []
        doc_id = 0
        
        for file_path in dataset_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    logger.info(f"Processing {file_path}...")
                    
                    if isinstance(data, list):
                        for item in data:
                            doc_text, metadata = self._process_document(item, file_path)
                            if doc_text:
                                documents.append(doc_text)
                                metadatas.append(metadata)
                                ids.append(f"doc_{doc_id}")
                                doc_id += 1
                                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Add documents to ChromaDB in batches
        if documents:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size] 
                batch_ids = ids[i:i+batch_size]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch_docs).tolist()
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings,
                    ids=batch_ids
                )
                
            logger.info(f"Successfully loaded {len(documents)} documents to vector store")
        else:
            logger.warning("No documents found to load")

    def _process_document(self, item: Dict, source_file: str) -> tuple:
        """Process individual document based on its structure"""
        try:
            metadata = {"source": source_file}
            
            if "question" in item and "response" in item:
                # QA format
                doc_text = f"Question: {item['question']}\nAnswer: {item['response']}"
                metadata["type"] = "qa"
                
            elif "Scheme Name" in item:
                # Scheme format
                doc_text = f"Scheme: {item['Scheme Name']}\n"
                if "Details" in item:
                    doc_text += f"Details: {item['Details']}\n"
                if "Benefits" in item:
                    doc_text += f"Benefits: {item['Benefits']}\n"
                if "Eligibility" in item:
                    doc_text += f"Eligibility: {item['Eligibility']}\n"
                if "Application Process" in item:
                    doc_text += f"Application Process: {item['Application Process']}\n"
                if "Documents Required" in item:
                    doc_text += f"Documents Required: {item['Documents Required']}\n"
                metadata["type"] = "scheme"
                metadata["scheme_name"] = item["Scheme Name"]
                
            elif "Scheme Title/Name" in item:
                # Tamil scheme format
                doc_text = f"Scheme: {item['Scheme Title/Name']}\n"
                if "Description" in item:
                    doc_text += f"Description: {item['Description']}\n"
                if "Concerned Department" in item:
                    doc_text += f"Department: {item['Concerned Department']}\n"
                if "Beneficiaries" in item:
                    doc_text += f"Beneficiaries: {item['Beneficiaries']}\n"
                if "Types of Benefits" in item:
                    doc_text += f"Benefits: {item['Types of Benefits']}\n"
                if "How To avail" in item:
                    doc_text += f"How to avail: {item['How To avail']}\n"
                metadata["type"] = "tamil_scheme"
                metadata["scheme_name"] = item["Scheme Title/Name"]
                
            elif "content" in item:
                # Content format
                doc_text = item["content"]
                metadata["type"] = "content"
                if "department" in item:
                    metadata["department"] = item["department"]
                    
            else:
                # Generic format
                doc_text = str(item)
                metadata["type"] = "generic"
                
            return doc_text, metadata
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None, None

    def translate_query_for_retrieval(self, query: str, language: str) -> str:
        """Translate Tamil query to English for better retrieval"""
        if language == "tamil":
            try:
                # Use LLM to translate Tamil query to English
                translation_prompt = f"""Translate the following Tamil query to English. Only provide the English translation, nothing else.

Tamil Query: {query}

English Translation:"""
                
                response = self.llm.invoke([HumanMessage(content=translation_prompt)])
                translated_query = response.content.strip()
                logger.info(f"Translated Tamil query: {query} -> {translated_query}")
                return translated_query
            except Exception as e:
                logger.error(f"Error translating query: {str(e)}")
                return query
        return query

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents for the query with enhanced relevance filtering"""
        try:
            # Detect language
            def detect_language(text):
                tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
                total_chars = len([c for c in text if c.isalpha()])
                return "tamil" if tamil_chars > total_chars * 0.3 else "english"
            
            query_language = detect_language(query)
            
            # Translate Tamil query to English for better retrieval
            search_query = self.translate_query_for_retrieval(query, query_language)
            
            # Generate query embedding using the (potentially translated) search query
            query_embedding = self.embedding_model.encode([search_query]).tolist()[0]
            
            # Search in ChromaDB with MANY results for comprehensive coverage
            # Retrieve more documents to ensure we ALWAYS have relevant information
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k * 4, 20)  # Increased to get maximum coverage
            )
            
            relevant_docs = []
            if results["documents"] and results["distances"]:
                docs = results["documents"][0]
                distances = results["distances"][0]
                
                # VERY lenient threshold to ensure we ALWAYS provide information
                # Lower distance = more relevant (cosine distance ranges from 0 to 2)
                for doc, distance in zip(docs, distances):
                    if distance < 1.5:  # Much more lenient threshold
                        relevant_docs.append(doc)
                
                # ALWAYS return documents - never leave empty
                if not relevant_docs and docs:
                    # If threshold filtering gives nothing, take top results
                    relevant_docs = docs[:min(len(docs), k + 3)]
                else:
                    # Return many documents to ensure comprehensive answers
                    relevant_docs = relevant_docs[:min(len(relevant_docs), k + 5)]
                
                # Final fallback - if still empty, take whatever we have
                if not relevant_docs:
                    relevant_docs = docs[:min(len(docs), 3)]
                
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def generate_response(self, query: str) -> str:
        """Generate response using RAG approach"""
        try:
            # Detect query language
            def detect_language(text):
                tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
                total_chars = len([c for c in text if c.isalpha()])
                return "tamil" if tamil_chars > total_chars * 0.3 else "english"
            
            query_language = detect_language(query)
            
            # Retrieve relevant documents - MAXIMIZE retrieval for comprehensive answers
            relevant_docs = self.retrieve_relevant_docs(query, k=10)
            
            # ALWAYS prepare context - never say we don't have information
            context = "\n\n".join(relevant_docs) if relevant_docs else ""
            context_available = len(relevant_docs) > 0
            
            # Get conversation history
            chat_history = self.memory.chat_memory.messages
            history_text = ""
            if chat_history:
                for msg in chat_history[-4:]:  # Last 2 exchanges
                    if isinstance(msg, HumanMessage):
                        history_text += f"User: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"Assistant: {msg.content}\n"
            
            # Language-specific instruction
            language_instruction = {
                "tamil": "USER IS ASKING IN TAMIL. YOU MUST RESPOND ENTIRELY IN TAMIL LANGUAGE ONLY. Do not use any English words except proper nouns if absolutely necessary.",
                "english": "USER IS ASKING IN ENGLISH. YOU MUST RESPOND ENTIRELY IN ENGLISH LANGUAGE ONLY. Do not use any Tamil words."
            }
            
            # Create prompt that ALWAYS provides helpful information
            prompt = f"""{self.system_prompt}

CONVERSATION HISTORY:
{history_text}

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{context}

CRITICAL INSTRUCTION - {language_instruction[query_language]}

USER QUERY: {query}

RESPONSE STRATEGY (MUST FOLLOW):
1. **PRIMARY APPROACH**: If the RELEVANT CONTEXT has information related to the query, use it to provide a comprehensive answer
2. **FALLBACK APPROACH**: If the specific scheme/service is not found in context BUT the query is clearly about Tamil Nadu government schemes/services, provide GENERAL guidance based on your knowledge:
   - Explain what TYPE of scheme/service this typically falls under
   - Mention which department(s) usually handle such matters
   - Provide GENERAL steps for accessing Tamil Nadu government schemes/services
   - Suggest where to find more specific information (e.g., official TN government portals, district offices)
   - Give contact information for relevant departments if known
3. **NEVER SAY**: "I don't have information" or "I cannot help" for ANY Tamil Nadu government scheme/service question
4. **ALWAYS BE HELPFUL**: Even with limited context, provide useful guidance and direction

RESPONSE REQUIREMENTS (MUST FOLLOW EXACTLY):
1. Language: Respond ONLY in {query_language.upper()} - no mixing, no translation
2. Always Helpful: NEVER refuse to answer questions about Tamil Nadu schemes/services - always provide useful information or guidance
3. Use Context First: Prioritize information from RELEVANT CONTEXT when available
4. General Knowledge: When context is limited, use your general knowledge of Tamil Nadu government structures and processes
5. Comprehensive: Provide detailed, thorough answers that truly help the user
6. Practical Guidance: Include actionable steps, contact information, and where to find more details
7. Professional Tone: Be confident and authoritative as an expert assistant
8. Structure: Organize information clearly with proper paragraphs and sections

ANSWERING FRAMEWORK:
- For GREETINGS/CASUAL: Keep brief (1-2 sentences), warm and friendly
- For SCHEME/SERVICE QUESTIONS with context: Provide detailed information from context
- For SCHEME/SERVICE QUESTIONS without specific context: Provide general guidance, relevant department info, typical process, and where to get specific details
- ALWAYS end with helpful next steps or contact information when relevant

CRITICAL: You are an EXPERT on Tamil Nadu government schemes and services. You MUST provide helpful, actionable information for EVERY query related to TN government schemes, services, certificates, or welfare programs. Use the context when available, and your expertise when context is limited.

Provide a helpful, comprehensive, and USEFUL response now:"""

            # Generate response
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
            
            # Update memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history"""
        history = []
        messages = self.memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "user": user_msg.content,
                    "assistant": ai_msg.content
                })
                
        return history

    def clear_conversation(self):
        """Clear conversation memory"""
        self.memory.clear()

def main():
    """Main Streamlit app with Premium ChatGPT-like UI"""
    st.set_page_config(
        page_title="TN Schemes ChatBot | தமிழ்நாடு திட்டங்கள்",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced Premium CSS styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main background with gradient */
        .stApp {
            background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        }
        
        /* Sidebar - FORCE VISIBILITY - Enhanced Dark Theme */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3) !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 21rem !important;
            min-width: 21rem !important;
            max-width: 21rem !important;
            position: relative !important;
            transform: translateX(0) !important;
            transition: none !important;
        }
        
        /* Force sidebar to stay expanded */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 21rem !important;
            min-width: 21rem !important;
            margin-left: 0 !important;
            transform: translateX(0) !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 21rem !important;
            min-width: 21rem !important;
        }
        
        /* Force all sidebar children to be visible */
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] > div > div,
        [data-testid="stSidebar"] * {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        /* Sidebar inner container */
        [data-testid="stSidebar"] > div:first-child {
            padding: 1rem !important;
            background: transparent !important;
        }
        
        /* Make collapse button visible but don't let it collapse */
        button[kind="header"][data-testid="baseButton-header"] {
            display: none !important;
        }
        
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        
        /* Sidebar text and markdown */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] div {
            color: #e8e8e8 !important;
            display: block !important;
            visibility: visible !important;
        }
        
        /* Sidebar Title */
        section[data-testid="stSidebar"] .sidebar-title {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1rem;
            letter-spacing: -0.5px;
        }
        
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 1.5rem;
        }
        
        /* Sidebar Buttons Base Style */
        section[data-testid="stSidebar"] .stButton button {
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
            letter-spacing: 0.3px;
            padding: 0.9rem 1.2rem;
        }
        
        /* New Chat Button - Premium Style */
        section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button,
        section[data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button:hover,
        section[data-testid="stSidebar"] button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button:active,
        section[data-testid="stSidebar"] button[kind="primary"]:active {
            transform: translateY(0);
        }
        
        /* Chat History Items - Sleek Cards */
        section[data-testid="stSidebar"] div[data-testid="stButton"]:not(:first-of-type) button {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e8e8e8;
            font-weight: 500;
            box-shadow: none;
            backdrop-filter: blur(10px);
            margin-bottom: 0.5rem;
            text-align: left;
            padding: 0.8rem 1rem;
        }
        
        section[data-testid="stSidebar"] div[data-testid="stButton"]:not(:first-of-type) button:hover {
            background: rgba(102, 126, 234, 0.15);
            border-color: rgba(102, 126, 234, 0.4);
            transform: translateX(4px);
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
        }
        
        /* Main Chat Area - Premium Layout with proper spacing */
        .main .block-container {
            padding: 2rem 2rem 10rem 2rem !important;
            max-width: 1000px;
            margin: 0 auto;
            min-height: calc(100vh - 12rem);
        }
        
        /* Chat messages container spacing */
        .main > div > div > div > div {
            margin-bottom: 2rem;
        }
        
        /* Header with Gradient */
        .main h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .main h3 {
            color: #a0a0a0;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 3rem;
        }
        
        /* Chat Messages Container - Messenger Style Layout */
        .stChatMessage {
            padding: 0 !important;
            margin: 1rem 0 !important;
            border: none !important;
            background: transparent !important;
            width: 100% !important;
            clear: both !important;
        }
        
        /* USER MESSAGE - RIGHT SIDE (like WhatsApp/Messenger) */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            display: block !important;
            text-align: right !important;
            width: 100% !important;
            padding-left: 20% !important;
            padding-right: 0 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {
            display: inline-flex !important;
            flex-direction: row-reverse !important;
            align-items: flex-start !important;
            justify-content: flex-end !important;
            max-width: 80% !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
            margin-left: 12px !important;
            margin-right: 0 !important;
            flex-shrink: 0 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div > div:last-child {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            padding: 12px 16px !important;
            border-radius: 18px 18px 4px 18px !important;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
            max-width: 100% !important;
            display: inline-block !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
            color: #ffffff !important;
            font-size: 15px !important;
            line-height: 1.6 !important;
            margin: 0 !important;
            background: transparent !important;
            padding: 0 !important;
            text-align: left !important;
        }
        
        /* ASSISTANT MESSAGE - LEFT SIDE (like WhatsApp/Messenger) */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            display: block !important;
            text-align: left !important;
            width: 100% !important;
            padding-right: 20% !important;
            padding-left: 0 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div {
            display: inline-flex !important;
            flex-direction: row !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
            max-width: 80% !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
            margin-right: 12px !important;
            margin-left: 0 !important;
            flex-shrink: 0 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div > div:last-child {
            background: rgba(255, 255, 255, 0.08) !important;
            color: #f0f0f0 !important;
            padding: 12px 16px !important;
            border-radius: 18px 18px 18px 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
            backdrop-filter: blur(10px) !important;
            max-width: 100% !important;
            display: inline-block !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
            color: #f0f0f0 !important;
            font-size: 15px !important;
            line-height: 1.7 !important;
            margin: 0.5rem 0 !important;
            background: transparent !important;
            padding: 0 !important;
            text-align: left !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p:first-child {
            margin-top: 0 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p:last-child {
            margin-bottom: 0 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) strong {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li {
            color: #f0f0f0 !important;
        }
        
        /* Chat Input - Modern Design with proper positioning */
        .stChatInputContainer {
            background: rgba(30, 30, 46, 0.95) !important;
            backdrop-filter: blur(20px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem 0 !important;
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            width: 100% !important;
            z-index: 999 !important;
            margin: 0 !important;
        }
        
        .stChatInput textarea {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            color: #ffffff !important;
            font-size: 16px !important;
            padding: 16px 20px !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        
        .stChatInput textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
            background: rgba(255, 255, 255, 0.08) !important;
        }
        
        .stChatInput textarea::placeholder {
            color: #888 !important;
        }
        
        /* Enhanced Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            border: 2px solid rgba(30, 30, 46, 0.8);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* Loading Spinner */
        .stSpinner > div {
            border-top-color: #667eea !important;
            border-right-color: #764ba2 !important;
        }
        
        /* Avatar Styling - Premium */
        [data-testid="chatAvatarIcon-user"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        [data-testid="chatAvatarIcon-assistant"] {
            background: linear-gradient(135deg, #10a37f 0%, #34d399 100%) !important;
            box-shadow: 0 4px 15px rgba(16, 163, 127, 0.4);
        }
        
        /* Expander Styling */
        .stExpander {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        
        .stExpander [data-testid="stMarkdownContainer"] {
            color: #e8e8e8;
        }
        
        /* Welcome Section */
        .welcome-section {
            text-align: center;
            padding: 4rem 2rem;
            animation: fadeIn 0.8s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .welcome-section h3 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .welcome-section p {
            color: #b0b0b0;
            font-size: 1.1rem;
        }
        
        /* Sample Question Buttons */
        .sample-questions .stButton button {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e8e8e8;
            font-weight: 500;
            box-shadow: none;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            border-radius: 10px;
            padding: 0.9rem 1.2rem;
            width: 100%;
        }
        
        .sample-questions .stButton button:hover {
            background: rgba(102, 126, 234, 0.2);
            border-color: rgba(102, 126, 234, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem 0.5rem 6rem 0.5rem;
            }
            
            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                padding-left: 5% !important;
            }
            
            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
                padding-right: 5% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = []
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing AI Assistant"):
            try:
                st.session_state.chatbot = TamilNaduSchemesRAGChatbot()
            except Exception as e:
                st.error(f"❌ Error initializing chatbot: {str(e)}")
                st.stop()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar with conversation history
    with st.sidebar:
        st.markdown('<p class="sidebar-title"><strong>SIVI AI</strong></p>', unsafe_allow_html=True)
        st.markdown("---")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat"):
            # Save current session if it has messages
            current_history = chatbot.get_conversation_history()
            if current_history:
                session_title = current_history[0]["user"][:45] + "..." if len(current_history[0]["user"]) > 45 else current_history[0]["user"]
                st.session_state.chat_sessions.insert(0, {
                    'id': len(st.session_state.chat_sessions),
                    'title': session_title,
                    'history': current_history.copy()
                })
            
            # Clear current chat
            chatbot.clear_conversation()
            st.session_state.current_session_id = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("RECENT CHATS")
        
        # Display chat history
        if st.session_state.chat_sessions:
            for idx, session in enumerate(st.session_state.chat_sessions[:10]):  # Show last 10
                if st.button(
                    f" {session['title']}", 
                    key=f"session_{session['id']}", 
                    use_container_width=True
                ):
                    # Load this session
                    chatbot.clear_conversation()
                    for exchange in session['history']:
                        chatbot.memory.chat_memory.add_user_message(exchange['user'])
                        chatbot.memory.chat_memory.add_ai_message(exchange['assistant'])
                    st.session_state.current_session_id = session['id']
                    st.rerun()
        else:
            st.markdown("*No chat history yet. Start a conversation!*")
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️About This AI"):
            st.markdown("""
            EXPERT AI ASSISTANT

            I have EXTENSIVE & COMPREHENSIVE knowledge of:

            ALL TN Government Schemes:
            • Financial assistance & subsidies
            • Educational scholarships & fellowships
            • Agricultural & farmer welfare schemes
            • Housing & construction loans
            • Employment & skill development
            • Pension schemes (old-age, widow, etc.)
            • Healthcare & medical assistance
            • Women & child welfare programs
            • SC/ST, minority & backward class schemes
            • Differently-abled welfare schemes
            
            ALL Government Services:
            • Birth, death, marriage certificates
            • Income, caste, community certificates
            • Land records (patta, chitta, survey)
            • Ration cards & family cards
            • Driving licenses & vehicle registration
            • Property & land registration
            • Business licenses & permits
            
            ALL Department Programs:
            • Revenue, Social Welfare, Agriculture
            • Education, Health, Labour, Transport
            • Rural & Urban Development
            • Police & other govt departments
            
            My Capabilities:
            -  **Bilingual Expert** (Tamil & English)
            -  **Complete Eligibility Information**
            -  **Detailed Benefits & Financial Aid**
            -  **Application Process Guidance**
            -  **Required Documents Lists**
            -  **Contact & Office Information**
            -  **Online & Offline Procedures**
            
             **Powered by:**
            - Advanced RAG Technology
            - ChromaDB Vector Search
            - AI LLM Integration
            
             **Tips:**
            - Ask me ANYTHING about TN schemes/services
            - I can answer in Tamil or English
            - I provide detailed, comprehensive answers
            - I'm your ULTIMATE TN govt schemes expert!
            """)
    
    # Main chat area
    st.title("SIVI AI")
    st.markdown("Your Intelligent Guide to TN Government Schemes & Services | தமிழ்நாடு அரசு திட்டங்கள் மற்றும் சேவைகளுக்கான உங்கள் நுண்ணறிவு வழிகாட்டி")

    # Display conversation
    history = chatbot.get_conversation_history()
    
    # Show welcome message if no history
    if not history:
        st.markdown("""
        <div class='welcome-section'>
            <h3>👋 Welcome to SIVI AI!</h3>
            <p>Tamilnadu Schemes and Services FAQ Chatbot</p>
            <p style='color: #888; font-size: 0.95rem; margin-top: 1rem;'>
                 I can help you with eligibility, benefits, application process, and more!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample questions for new chat
        st.markdown("<b>Suggestions:</b>", unsafe_allow_html=True)
        st.markdown("<div class='sample-questions'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Student Scholarships Available?", use_container_width=True, key="q1"):
                st.session_state.sample_query = "What scholarships are available for students in Tamil Nadu?"
                st.rerun()
            if st.button("Housing Scheme Details?", use_container_width=True, key="q2"):
                st.session_state.sample_query = "How can I apply for housing schemes in Tamil Nadu?"
                st.rerun()
            if st.button("Farmer Welfare Schemes?", use_container_width=True, key="q3"):
                st.session_state.sample_query = "What are the welfare schemes available for farmers?"
                st.rerun()
        
        with col2:
            if st.button("விவசாய திட்டங்கள்?", use_container_width=True, key="q4"):
                st.session_state.sample_query = "விவசாய திட்டங்கள் என்னென்ன உள்ளன?"
                st.rerun()
            if st.button("வருமான சான்றிதழ்?", use_container_width=True, key="q5"):
                st.session_state.sample_query = "வருமான சான்றிதழ் பெற என்ன ஆவணங்கள் தேவை?"
                st.rerun()
            if st.button("நிதி உதவி திட்டங்கள்?", use_container_width=True, key="q6"):
                st.session_state.sample_query = "தமிழ்நாட்டில் என்னென்ன நிதி உதவி திட்டங்கள் உள்ளன?"
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chat messages using custom HTML for proper alignment
    for exchange in history:
        # User message - right aligned
        st.markdown(f"""
            <div style='text-align: right; margin: 10px 0; clear: both;'>
                <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 70%; text-align: left; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);'>
                    {exchange["user"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Bot message - left aligned
        st.markdown(f"""
            <div style='text-align: left; margin: 10px 0; clear: both;'>
                <div style='display: inline-block; background: rgba(255, 255, 255, 0.08); color: #f0f0f0; padding: 12px 16px; border-radius: 18px 18px 18px 4px; max-width: 70%; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);'>
                    {exchange["assistant"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Handle sample query
    if hasattr(st.session_state, 'sample_query'):
        user_query = st.session_state.sample_query
        delattr(st.session_state, 'sample_query')
        
        # Display user message - right aligned
        st.markdown(f"""
            <div style='text-align: right; margin: 10px 0; clear: both;'>
                <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 70%; text-align: left; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);'>
                    {user_query}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate and display bot response - left aligned
        with st.spinner("🧠 Thinking..."):
            response = chatbot.generate_response(user_query)
        
        st.markdown(f"""
            <div style='text-align: left; margin: 10px 0; clear: both;'>
                <div style='display: inline-block; background: rgba(255, 255, 255, 0.08); color: #f0f0f0; padding: 12px 16px; border-radius: 18px 18px 18px 4px; max-width: 70%; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);'>
                    {response}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.rerun()
    
    # Chat input
    user_query = st.chat_input("Ask anything | எதையும் கேளுங்கள்")
    
    if user_query:
        # Display user message - right aligned
        st.markdown(f"""
            <div style='text-align: right; margin: 10px 0; clear: both;'>
                <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; max-width: 70%; text-align: left; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);'>
                    {user_query}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate and display bot response - left aligned
        with st.spinner("🧠 Thinking..."):
            response = chatbot.generate_response(user_query)
        
        st.markdown(f"""
            <div style='text-align: left; margin: 10px 0; clear: both;'>
                <div style='display: inline-block; background: rgba(255, 255, 255, 0.08); color: #f0f0f0; padding: 12px 16px; border-radius: 18px 18px 18px 4px; max-width: 70%; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);'>
                    {response}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.rerun()

if __name__ == "__main__":
    main()
