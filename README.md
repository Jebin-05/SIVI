# Tamil Nadu Government Schemes Bilingual FAQ Chatbot

## Overview | கண்ணோட்டம்

This is a comprehensive bilingual FAQ chatbot for Tamil Nadu Government Schemes that provides accurate information about various government schemes, services, and benefits available to citizens of Tamil Nadu. The chatbot supports both **Tamil** and **English** languages and uses advanced RAG (Retrieval-Augmented Generation) architecture.

**தமிழ்நாடு அரசு திட்டங்களுக்கான இரு மொழி FAQ சாட்பாட்** - தமிழ்நாடு குடிமக்களுக்கு கிடைக்கும் பல்வேறு அரசு திட்டங்கள், சேவைகள் மற்றும் நன்மைகள் பற்றிய துல்லியமான தகவல்களை வழங்குகிறது.

## Features | அம்சங்கள்

### English Features:
- **Bilingual Support**: Seamlessly switch between Tamil and English
- **Comprehensive Coverage**: Information about 1000+ government schemes
- **RAG Architecture**: Advanced retrieval system for accurate responses
- **Conversation Memory**: Maintains context throughout the conversation
- **Real-time Processing**: Instant responses to user queries
- **User-friendly Interface**: Clean Streamlit web interface

### தமிழ் அம்சங்கள்:
- **இரு மொழி ஆதரவு**: தமிழ் மற்றும் ஆங்கிலம் இடையே எளிதாக மாறவும்
- **விரிவான கவரேஜ்**: 1000+ அரசு திட்டங்கள் பற்றிய தகவல்கள்
- **RAG கட்டமைப்பு**: துல்லியமான பதில்களுக்கான மேம்பட்ட மீட்டெடுப்பு அமைப்பு
- **உரையாடல் நினைவகம்**: உரையாடல் முழுவதும் சூழலை பராமரிக்கிறது
- **நிகழ்நேர செயலாக்கம்**: பயனர் கேள்விகளுக்கு உடனடி பதில்கள்
- **பயனர் நட்பு இடைமுகம்**: சுத்தமான Streamlit வலை இடைமுகம்

## Technology Stack | தொழில்நுட்ப அடுக்கு

- **LangChain**: For conversation management and RAG implementation
- **ChromaDB**: Vector database for efficient document storage and retrieval
- **Sentence Transformers**: For generating semantic embeddings
- **OpenRouter API**: LLM integration with Gemma model
- **Streamlit**: Web interface for the chatbot
- **Python**: Core programming language

## Dataset Coverage | தரவுத்தொகுப்பு கவரேஜ்

The chatbot contains information from multiple comprehensive datasets:

1. **finetune_QA.json** (21,662 entries): Q&A pairs about financial schemes
2. **processed_rag_dept.json**: Department-specific government orders
3. **processed_rag_services.json** (142 entries): Citizen services information
4. **rag_new_scheme.json** (1,882 entries): Latest scheme details
5. **tamil_scheme_data.json** (1,564 entries): Tamil language scheme data

### Scheme Categories Covered:
- **Agricultural Schemes** | **விவசாய திட்டங்கள்**
- **Educational Scholarships** | **கல்வி உதவித்தொகைகள்**
- **Housing & Construction** | **வீட்டுவசதி மற்றும் கட்டுமானம்**
- **Employment & Skills** | **வேலைவாய்ப்பு மற்றும் திறன்கள்**
- **Welfare Schemes** | **நலன்புரி திட்டங்கள்**
- **Certificate Services** | **சான்றிதழ் சேவைகள்**
- **Financial Assistance** | **நிதி உதவி**
- **Transport Services** | **போக்குவரத்து சேவைகள்**

## Installation & Setup | நிறுவல் மற்றும் அமைப்பு

### Prerequisites | முன்நிபந்தனைகள்:
- Python 3.8 or higher
- Virtual environment support
- Internet connection for model downloads

### Quick Start | விரைவு தொடக்கம்:

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/jebin/Desktop/Schemes_Chatbot
   ```

2. **Run the setup script:**
   ```bash
   ./run_chatbot.sh
   ```

3. **Manual setup (alternative):**
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Install packages
   pip install langchain langchain-community chromadb sentence-transformers openai langchain-openai python-dotenv streamlit transformers torch numpy pandas requests
   
   # Run the chatbot
   streamlit run tn_schemes_chatbot.py
   ```

## Usage Examples | பயன்பாட்டு எடுத்துக்காட்டுகள்

### English Queries:
- "What agricultural schemes are available for farmers?"
- "How to apply for housing loan from cooperative banks?"
- "What documents are required for income certificate?"
- "Tell me about scholarships for SC/ST students"
- "What are the benefits of Poompuhar State Award?"

### Tamil Queries | தமிழ் கேள்விகள்:
- "விவசாயிகளுக்கு என்ன திட்டங்கள் உள்ளன?"
- "கூட்டுறவு வங்கியில் வீட்டு கடனுக்கு எப்படி விண்ணப்பிப்பது?"
- "வருமான சான்றிதழுக்கு என்ன ஆவணங்கள் தேவை?"
- "எஸ்சி/எஸ்டி மாணவர்களுக்கான உதவித்தொகை பற்றி சொல்லுங்கள்"
- "பூம்புகார் மாநில விருதின் நன்மைகள் என்ன?"

## Features in Detail | விரிவான அம்சங்கள்

### 1. Advanced RAG Architecture:
- **Vector Search**: Uses sentence transformers for semantic similarity
- **Context Retrieval**: Retrieves top-3 most relevant documents
- **Response Generation**: Uses OpenRouter's Gemma model for accurate responses

### 2. Conversation Memory:
- **Buffer Memory**: Maintains recent conversation context
- **Contextual Responses**: Understands follow-up questions
- **Session Persistence**: Remembers conversation within session

### 3. Bilingual Intelligence:
- **Language Detection**: Automatically detects user's preferred language
- **Consistent Response**: Responds in the same language as the query
- **Cultural Context**: Understands Tamil cultural and administrative context

### 4. Comprehensive System Prompt:
- **50+ lines of detailed instructions** for the AI assistant
- **Strict adherence** to Tamil Nadu government schemes content
- **Professional tone** and accurate information delivery
- **Clear limitations** and verification recommendations

## Configuration | உள்ளமைவு

### Environment Variables:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: Model name (default: google/gemma-3n-e4b-it:free)

### Customization Options:
- **Embedding Model**: Change sentence transformer model
- **Vector Database**: Modify ChromaDB settings
- **Response Length**: Adjust max_tokens parameter
- **Retrieval Count**: Modify number of documents retrieved

## Project Structure | திட்ட கட்டமைப்பு

```
Schemes_Chatbot/
├── tn_schemes_chatbot.py      # Main chatbot application
├── run_chatbot.sh             # Setup and run script
├── .env                       # Environment variables
├── .venv/                     # Virtual environment
├── chroma_db/                 # Vector database storage
└── Datasets/                  # Data files
    ├── finetune_QA.json
    ├── processed_rag_dept.json
    ├── processed_rag_services.json
    ├── rag_new_scheme.json
    └── tamil_scheme_data.json
```

## Performance | செயல்திறன்

- **Response Time**: < 3 seconds for most queries
- **Accuracy**: High accuracy based on official government data
- **Memory Usage**: Optimized for efficient memory consumption
- **Scalability**: Can handle multiple concurrent users

## Troubleshooting | சிக்கல் தீர்வு

### Common Issues:

1. **ChromaDB not initializing:**
   - Delete `chroma_db` folder and restart

2. **OpenRouter API errors:**
   - Check API key in `.env` file
   - Verify internet connection

3. **Model loading issues:**
   - Ensure sufficient disk space for model downloads
   - Check internet connectivity

## Contributing | பங்களித்தல்

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## Support | ஆதரவு

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check documentation for common solutions

## License | உரிமம்

This project is created for educational and public service purposes to help Tamil Nadu citizens access government scheme information easily.

---

**Disclaimer | மறுப்பு:** This chatbot provides information based on available datasets. Users are advised to verify latest information from official government sources before making any applications or decisions.

**अस्वीकरण:** यह चैटबॉट उपलब्ध डेटासेट के आधार पर जानकारी प्रदान करता है। उपयोगकर्ताओं को सलाह दी जाती है कि वे कोई भी आवेदन या निर्णय लेने से पहले आधिकारिक सरकारी स्रोतों से नवीनतम जानकारी सत्यापित करें।
