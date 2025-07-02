import os
import warnings
import sys

# Comprehensive environment setup to suppress all warnings and errors
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress ChromaDB telemetry completely
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'

import sqlite3
import csv
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import json
import numpy as np
import re
from flask import Flask, request, jsonify, render_template, url_for, redirect, session, flash, send_from_directory
from markupsafe import Markup
from flask_cors import CORS

# Suppress TensorFlow warnings properly
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Import markdown renderer with FIXED preset
try:
    from markdown_it import MarkdownIt
    md = MarkdownIt('gfm-like')  # FIXED: Changed from 'gfm' to 'gfm-like'
    MARKDOWN_AVAILABLE = True
except ImportError:
    print("Warning: markdown-it-py not installed. Install with: pip install markdown-it-py")
    MARKDOWN_AVAILABLE = False
except Exception as e:
    print(f"Warning: MarkdownIt initialization failed: {e}")
    print("Falling back to basic markdown conversion")
    MARKDOWN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
CORS(app, supports_credentials=True, origins=['http://localhost:5000', 'http://127.0.0.1:5000'])

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
PDF_FOLDER = os.path.join(UPLOAD_FOLDER, 'pdf')
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
ELECTRICAL_PDF_FOLDER = os.path.join(UPLOAD_FOLDER, 'electrical_pdf')
ELECTRICAL_IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'electrical_images')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('electrical_data', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(ELECTRICAL_PDF_FOLDER, exist_ok=True)
os.makedirs(ELECTRICAL_IMAGE_FOLDER, exist_ok=True)
os.makedirs('user_data', exist_ok=True)

# CSV file paths
USERS_CSV = 'user_data/users.csv'
CHAT_LOGS_CSV = 'user_data/chat_logs.csv'
ELECTRICAL_CHAT_LOGS_CSV = 'user_data/electrical_chat_logs.csv'

def md_to_html(text: str) -> str:
    """
    Convert chatbot Markdown (supports tables) to sanitized HTML.
    """
    if MARKDOWN_AVAILABLE:
        try:
            html_content = md.render(text)
            return Markup(html_content)  # Mark as safe HTML
        except Exception as e:
            print(f"Error rendering markdown: {e}")
            return Markup(basic_markdown_to_html(text))
    else:
        # Fallback: basic manual conversion if markdown-it-py not available
        return Markup(basic_markdown_to_html(text))

def basic_markdown_to_html(text: str) -> str:
    """
    Enhanced fallback markdown to HTML converter for tables and formatting
    """
    # Convert headers first
    text = re.sub(r'^## (.*$)', r'<h2 style="color: #333; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #667eea;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*$)', r'<h3 style="color: #495057; margin: 1.2rem 0 0.8rem 0;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^#### (.*$)', r'<h4 style="color: #666; margin: 1rem 0 0.6rem 0;">\1</h4>', text, flags=re.MULTILINE)
    
    # Convert bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Convert code blocks
    text = re.sub(r'``````', r'<pre style="background: #f8f9fa; padding: 1rem; border-radius: 5px; overflow-x: auto; border-left: 4px solid #667eea; margin: 1rem 0;"><code>\1</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*?)`', r'<code style="background: #f8f9fa; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: monospace;">\1</code>', text)
    
    # Process tables
    lines = text.split('\n')
    in_table = False
    table_lines = []
    result_lines = []
    
    for i, line in enumerate(lines):
        # Check if line looks like a table row
        if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
        elif in_table:
            # Process the table we've accumulated
            if table_lines:
                result_lines.append(convert_table_to_html(table_lines))
                table_lines = []
            in_table = False
            result_lines.append(line)
        else:
            result_lines.append(line)
    
    # Handle table at end of text
    if in_table and table_lines:
        result_lines.append(convert_table_to_html(table_lines))
    
    # Convert lists
    text = '\n'.join(result_lines)
    text = re.sub(r'^- (.*$)', r'<li style="margin: 0.3rem 0;">\1</li>', text, flags=re.MULTILINE)
    
    # Wrap standalone <li> elements in <ul>
    text = re.sub(r'(<li[^>]*>.*?</li>)', r'<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">\1</ul>', text)
    
    # Convert line breaks but preserve HTML structure
    text = re.sub(r'\n(?![<])', '<br>', text)
    
    return text

def convert_table_to_html(table_lines):
    """Convert markdown table lines to professional HTML table"""
    if len(table_lines) < 1:
        return '\n'.join(table_lines)
    
    # Enhanced table styling
    table_style = '''
    border-collapse: collapse; 
    width: 100%; 
    margin: 1.5rem 0; 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
    '''
    
    html = f'<table style="{table_style}">'
    
    # Process header row
    header_line = table_lines[0].strip()
    if header_line.startswith('|') and header_line.endswith('|'):
        header_cells = [cell.strip() for cell in header_line[1:-1].split('|')]
        html += '<thead>'
        html += '<tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">'
        for cell in header_cells:
            html += f'''<th style="
                border: none; 
                padding: 1rem 0.75rem; 
                color: white; 
                font-weight: 600; 
                text-align: left;
                font-size: 0.9rem;
            ">{cell}</th>'''
        html += '</tr>'
        html += '</thead>'
    
    # Find separator line and data start
    data_start = 1
    for i in range(1, len(table_lines)):
        if '---' in table_lines[i] or '--' in table_lines[i]:
            data_start = i + 1
            break
    
    # Process data rows
    html += '<tbody>'
    row_count = 0
    for line in table_lines[data_start:]:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            cells = [cell.strip() for cell in line[1:-1].split('|')]
            row_bg = '#f8f9fa' if row_count % 2 == 0 else 'white'
            html += f'<tr style="background: {row_bg};" onmouseover="this.style.background=\'#e9ecef\'" onmouseout="this.style.background=\'{row_bg}\'">'
            for cell in cells:
                html += f'''<td style="
                    border: none; 
                    padding: 0.75rem; 
                    border-bottom: 1px solid #dee2e6;
                    font-size: 0.85rem;
                    color: #495057;
                ">{cell}</td>'''
            html += '</tr>'
            row_count += 1
    html += '</tbody></table>'
    
    return html

def init_csv_files():
    """Initialize CSV files with proper encoding"""
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['user_id', 'username', 'email', 'password_hash', 'full_name', 'registration_date', 'last_login'])

    if not os.path.exists(CHAT_LOGS_CSV):
        with open(CHAT_LOGS_CSV, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['log_id', 'user_id', 'username', 'user_message', 'bot_response', 'timestamp', 'session_id'])

    if not os.path.exists(ELECTRICAL_CHAT_LOGS_CSV):
        with open(ELECTRICAL_CHAT_LOGS_CSV, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['log_id', 'user_id', 'username', 'user_message', 'bot_response', 'timestamp', 'session_id'])

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    if hashed is None:
        return False
    cleaned_hash = str(hashed).strip()
    input_hash = hash_password(password)
    return input_hash == cleaned_hash

def get_user_by_username_or_email(identifier):
    """Get user data from CSV by username or email"""
    try:
        with open(USERS_CSV, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                cleaned_row = {}
                for key, value in row.items():
                    if value is not None:
                        cleaned_row[key] = str(value).strip()
                    else:
                        cleaned_row[key] = value
                
                if (cleaned_row['username'] == identifier.strip() or 
                    cleaned_row['email'] == identifier.strip()):
                    return cleaned_row
        return None
    except (FileNotFoundError, Exception):
        return None

def create_user(username, email, password, full_name):
    """Create new user with proper CSV handling"""
    try:
        user_id = datetime.now().strftime('%Y%m%d%H%M%S') + str(hash(username))[-4:]
        password_hash = hash_password(password)
        registration_date = datetime.now().isoformat()
        
        with open(USERS_CSV, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                user_id.strip(),
                username.strip(),
                email.strip(),
                password_hash.strip(),
                full_name.strip(),
                registration_date,
                ''
            ])
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

def update_last_login(username):
    """Update user's last login time"""
    try:
        users = []
        with open(USERS_CSV, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'].strip() == username.strip():
                    row['last_login'] = datetime.now().isoformat()
                users.append(row)

        with open(USERS_CSV, 'w', newline='', encoding='utf-8') as file:
            if users:
                fieldnames = users[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(users)
    except Exception as e:
        print(f"Error updating last login: {e}")

def log_chat_interaction(user_id, username, user_message, bot_response, session_id, chat_type='material'):
    """Log chat interaction to CSV"""
    try:
        log_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
        timestamp = datetime.now().isoformat()
        
        csv_file = CHAT_LOGS_CSV if chat_type == 'material' else ELECTRICAL_CHAT_LOGS_CSV
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([log_id, user_id, username, user_message, bot_response, timestamp, session_id])
    except Exception as e:
        print(f"Error logging chat interaction: {e}")

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def analyze_image_for_compliance(image_path):
    """Basic image analysis to check material compliance"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            avg_r, avg_g, avg_b = np.mean(r), np.mean(g), np.mean(b)
        else:
            avg_r = avg_g = avg_b = np.mean(img_array)
        
        compliance_issues = []
        if avg_r > 200 and avg_g < 50 and avg_b < 50:
            compliance_issues.append("Possible rust or oxidation detected")
        
        if np.std(img_array) < 20:
            compliance_issues.append("Low contrast may indicate surface uniformity issues")
        
        return {
            "compliant": len(compliance_issues) == 0,
            "issues": compliance_issues,
            "metadata": {
                "dimensions": img.size,
                "format": img.format,
                "color_averages": {
                    "red": float(avg_r),
                    "green": float(avg_g),
                    "blue": float(avg_b)
                }
            }
        }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return {"error": str(e), "compliant": False, "issues": ["Analysis failed"]}

def login_required(f):
    """Decorator to require login for protected routes"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_database():
    conn = sqlite3.connect('material_specification_chat.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS electrical_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_messages INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS electrical_user_sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_messages INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            analysis_result TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS electrical_uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            analysis_result TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def get_enhanced_prompt_template():
    return '''You are a material specification identification expert assistant.

**RESPONSE FORMAT INSTRUCTIONS:**
- Always use markdown formatting in your responses
- Structure responses with clear headers (##, ###)
- Use **bold** for important specifications and standards
- Use bullet points (-) for lists and requirements
- Use code blocks (```
- Create tables when comparing materials or specifications using proper markdown table syntax
- Use numbered lists (1., 2., 3.) for procedures and processes

**TABLE FORMAT EXAMPLE:**
| Material | Grade | Tensile Strength | Application |
|----------|-------|------------------|-------------|
| Steel | ASTM A36 | 400-550 MPa | Structural |
| Aluminum | 6061-T6 | 310 MPa | Aerospace |

**CONTENT GUIDELINES:**
1. Only answer questions about material specifications, procurement, and supply chain management
2. If asked about other topics, respond: "## I specialize in material specifications\\n\\nI focus on material identification and procurement. How can I help you with material-related inquiries?"
3. Help identify materials based on descriptions, specifications, grades, and standards
4. Provide information about material properties, certifications, and compliance requirements

Use this context to provide accurate material information:
{context}

Question: {question}

**Expert Material Specification Answer (in markdown format):**'''

def get_electrical_prompt_template():
    return '''You are an electrical specification identification expert assistant.

**RESPONSE FORMAT INSTRUCTIONS:**
- Always use markdown formatting in your responses
- Structure responses with clear headers (##, ###)
- Use **bold** for important electrical specifications and standards
- Use bullet points (-) for lists and requirements
- Use code blocks (```) for electrical standards and specifications
- Create tables when comparing electrical components or specifications using proper markdown table syntax
- Use numbered lists (1., 2., 3.) for procedures and processes

**TABLE FORMAT EXAMPLE:**
| Component | Rating | Standard | Application |
|-----------|--------|----------|-------------|
| Circuit Breaker | 480V, 100A | IEC 60947 | Motor Control |
| Transformer | 11kV/480V | IEEE C57 | Distribution |

**CONTENT GUIDELINES:**
1. Only answer questions about electrical specifications, electrical engineering, power systems, and electrical safety
2. If asked about other topics, respond: "## I specialize in electrical specifications\\n\\nI focus on electrical engineering and specifications. How can I help you with electrical-related inquiries?"
3. Help identify electrical components based on descriptions, specifications, ratings, and standards
4. Provide information about electrical properties, safety requirements, and compliance standards
5. Support electrical system design, power calculations, and equipment selection

Use this context to provide accurate electrical information:
{context}

Question: {question}

**Expert Electrical Specification Answer (in markdown format):**'''

def initialize_components():
    try:
        documents = []
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
        except Exception as e:
            print(f"Warning: Could not load PDF files: {e}")

        # Load text files
        try:
            txt_loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader, silent_errors=True)
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
        except Exception as e:
            print(f"Warning: Could not load text files: {e}")

        # Load CSV files
        try:
            csv_loader = DirectoryLoader("data", glob="*.csv", loader_cls=CSVLoader, silent_errors=True)
            csv_docs = csv_loader.load()
            documents.extend(csv_docs)
        except Exception as e:
            print(f"Warning: Could not load CSV files: {e}")

        # If no documents loaded, create a fallback
        if not documents:
            print("Warning: No documents found in data directory")
            from langchain.schema import Document
            documents = [Document(page_content="Material specification database", metadata={"source": "fallback"})]

        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create embeddings with full compatibility
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # Create ChromaDB with telemetry completely disabled
        import chromadb
        from chromadb.config import Settings

        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            persist_directory="./chroma_db_material"
        )

        vector_db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory="./chroma_db_material",
            client_settings=chroma_settings
        )

        # Create LLM
        llm = ChatGroq(
            temperature=0,
            groq_api_key="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",
            model_name="llama-3.3-70b-versatile"
        )

        # ENHANCED PROMPT WITH MARKDOWN FORMATTING
        PROMPT = PromptTemplate(
            template=get_enhanced_prompt_template(),
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

    except Exception as e:
        print(f"Error initializing components: {e}")
        return None

def initialize_electrical_components():
    try:
        documents = []
        
        # Load electrical PDF files
        try:
            pdf_loader = DirectoryLoader("electrical_data", glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
        except Exception as e:
            print(f"Warning: Could not load electrical PDF files: {e}")

        # Load electrical text files
        try:
            txt_loader = DirectoryLoader("electrical_data", glob="*.txt", loader_cls=TextLoader, silent_errors=True)
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
        except Exception as e:
            print(f"Warning: Could not load electrical text files: {e}")

        # Load electrical CSV files
        try:
            csv_loader = DirectoryLoader("electrical_data", glob="*.csv", loader_cls=CSVLoader, silent_errors=True)
            csv_docs = csv_loader.load()
            documents.extend(csv_docs)
        except Exception as e:
            print(f"Warning: Could not load electrical CSV files: {e}")

        # If no documents loaded, create a fallback
        if not documents:
            print("Warning: No electrical documents found in electrical_data directory")
            from langchain.schema import Document
            documents = [Document(page_content="Electrical specification database", metadata={"source": "electrical_fallback"})]

        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create embeddings with full compatibility
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # Create ChromaDB with telemetry completely disabled
        import chromadb
        from chromadb.config import Settings

        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            persist_directory="./chroma_db_electrical"
        )

        vector_db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory="./chroma_db_electrical",
            client_settings=chroma_settings
        )

        # Create LLM (same API key)
        llm = ChatGroq(
            temperature=0,
            groq_api_key="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",
            model_name="llama-3.3-70b-versatile"
        )

        # ELECTRICAL PROMPT WITH MARKDOWN FORMATTING
        PROMPT = PromptTemplate(
            template=get_electrical_prompt_template(),
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

    except Exception as e:
        print(f"Error initializing electrical components: {e}")
        return None

def validate_response(response):
    if isinstance(response, dict) and "result" in response:
        result = response["result"]
    elif isinstance(response, dict):
        result = str(response)
    else:
        result = str(response)
    
    if not result or not result.strip():
        result = "Sorry, I couldn't generate a valid answer. Please try again."
    
    return result.strip()

def store_conversation(session_id, user_message, bot_response, response_time, conversation_type='material'):
    try:
        conn = sqlite3.connect('material_specification_chat.db')
        cursor = conn.cursor()
        
        table_name = 'conversations' if conversation_type == 'material' else 'electrical_conversations'
        sessions_table = 'user_sessions' if conversation_type == 'material' else 'electrical_user_sessions'
        
        cursor.execute(f'''
            INSERT INTO {table_name} (session_id, user_message, bot_response, response_time)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, response_time))
        
        cursor.execute(f'''
            INSERT OR REPLACE INTO {sessions_table} (session_id, last_active, total_messages)
            VALUES (?, CURRENT_TIMESTAMP,
                    COALESCE((SELECT total_messages FROM {sessions_table} WHERE session_id = ?), 0) + 1)
        ''', (session_id, session_id))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error storing conversation: {e}")

def get_conversation_history(session_id, limit=10, conversation_type='material'):
    try:
        conn = sqlite3.connect('material_specification_chat.db')
        cursor = conn.cursor()
        
        table_name = 'conversations' if conversation_type == 'material' else 'electrical_conversations'
        
        cursor.execute(f'''
            SELECT user_message, bot_response, timestamp
            FROM {table_name}
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        history = cursor.fetchall()
        conn.close()
        return list(reversed(history))
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def get_uploaded_files_count(file_type='material'):
    """Get count of uploaded files with proper error handling"""
    try:
        conn = sqlite3.connect('material_specification_chat.db')
        cursor = conn.cursor()
        
        table_name = 'uploaded_files' if file_type == 'material' else 'electrical_uploaded_files'
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 0
    except Exception as e:
        print(f"Error getting uploaded files count: {e}")
        return 0

# Initialize everything
init_csv_files()
init_database()
qa_chain = initialize_components()
electrical_qa_chain = initialize_electrical_components()

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['username'].strip()
        password = request.form['password']
        
        user = get_user_by_username_or_email(identifier)
        if user and verify_password(password, user['password_hash']):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            update_last_login(user['username'])
            return redirect(url_for('index'))
        else:
            flash('Invalid username/email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        full_name = request.form['full_name'].strip()
        
        if get_user_by_username_or_email(username) or get_user_by_username_or_email(email):
            flash('Username or email already exists')
            return render_template('register.html')
        
        if create_user(username, email, password, full_name):
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Please try again.')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Protected Routes
@app.route('/')
@login_required
def index():
    return render_template('material_index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files['file']
        session_id = request.form.get('session_id', 'default_session')

        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_type = filename.rsplit('.', 1)[1].lower()

            if file_type == 'pdf':
                save_folder = PDF_FOLDER
            else:
                save_folder = IMAGE_FOLDER

            file_path = os.path.join(save_folder, filename)
            file.save(file_path)

            analysis_result = {}
            if file_type == 'pdf':
                extracted_text = extract_text_from_pdf(file_path)
                if not extracted_text or len(extracted_text) < 50:
                    analysis_result = {"warning": "The PDF contains very little text or could not be processed"}
                else:
                    analysis_result = {"status": "PDF processed successfully", "text_length": len(extracted_text)}
            else:
                analysis_result = analyze_image_for_compliance(file_path)

            conn = sqlite3.connect('material_specification_chat.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO uploaded_files (session_id, file_name, file_path, file_type, analysis_result)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, filename, file_path, file_type, json.dumps(analysis_result)))
            conn.commit()
            conn.close()

            return jsonify({
                "success": True,
                "file_name": filename,
                "file_type": file_type,
                "analysis_result": analysis_result
            })

        return jsonify({"success": False, "error": "File type not allowed"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/electrical/upload', methods=['POST'])
@login_required
def upload_electrical_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files['file']
        session_id = request.form.get('session_id', 'default_electrical_session')

        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_type = filename.rsplit('.', 1)[1].lower()

            if file_type == 'pdf':
                save_folder = ELECTRICAL_PDF_FOLDER
            else:
                save_folder = ELECTRICAL_IMAGE_FOLDER

            file_path = os.path.join(save_folder, filename)
            file.save(file_path)

            analysis_result = {}
            if file_type == 'pdf':
                extracted_text = extract_text_from_pdf(file_path)
                if not extracted_text or len(extracted_text) < 50:
                    analysis_result = {"warning": "The PDF contains very little text or could not be processed"}
                else:
                    analysis_result = {"status": "Electrical PDF processed successfully", "text_length": len(extracted_text)}
            else:
                analysis_result = analyze_image_for_compliance(file_path)

            conn = sqlite3.connect('material_specification_chat.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO electrical_uploaded_files (session_id, file_name, file_path, file_type, analysis_result)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, filename, file_path, file_type, json.dumps(analysis_result)))
            conn.commit()
            conn.close()

            return jsonify({
                "success": True,
                "file_name": filename,
                "file_type": file_type,
                "analysis_result": analysis_result
            })

        return jsonify({"success": False, "error": "File type not allowed"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    start_time = datetime.now()
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default_session")

        if not user_message:
            return jsonify({"bot_response": "Please enter a valid material specification question"})

        file_references = re.findall(r"file:([a-zA-Z0-9_\-\.]+)", user_message)
        if file_references:
            conn = sqlite3.connect('material_specification_chat.db')
            cursor = conn.cursor()
            for file_ref in file_references:
                cursor.execute('''
                    SELECT file_path, file_type, analysis_result
                    FROM uploaded_files
                    WHERE file_name = ? AND session_id = ?
                ''', (file_ref, session_id))
                file_info = cursor.fetchone()
                if file_info:
                    file_path, file_type, analysis_result = file_info
                    user_message += f"\n[Analysis of {file_ref}: {analysis_result}]"
            conn.close()

        if qa_chain:
            result = qa_chain.invoke({"query": user_message})
            raw_bot_response = validate_response(result)
            # Convert markdown to HTML with proper Flask Markup
            html_bot_response = md_to_html(raw_bot_response)
        else:
            html_bot_response = Markup("I'm having trouble accessing the knowledge base. Please try again.")

        response_time = (datetime.now() - start_time).total_seconds()
        store_conversation(session_id, user_message, raw_bot_response, response_time, 'material')
        
        log_chat_interaction(
            session.get('user_id'),
            session.get('username'),  
            user_message,
            raw_bot_response,
            session_id,
            'material'
        )

        return jsonify({
            "bot_response": str(html_bot_response),  # Convert Markup to string for JSON
            "session_id": session_id,
            "response_time": response_time,
            "citations": []
        })

    except Exception as e:
        error_response = f"Error processing material specification request: {str(e)}"
        store_conversation(session_id, user_message, error_response, 0, 'material')
        return jsonify({"bot_response": error_response})

@app.route("/electrical/chat", methods=["POST"])
@app.route("/api/electrical/chat", methods=["POST"])
@login_required
def electrical_chat():
    start_time = datetime.now()
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default_electrical_session")

        if not user_message:
            return jsonify({"bot_response": "Please enter a valid electrical specification question"})

        file_references = re.findall(r"file:([a-zA-Z0-9_\-\.]+)", user_message)
        if file_references:
            conn = sqlite3.connect('material_specification_chat.db')
            cursor = conn.cursor()
            for file_ref in file_references:
                cursor.execute('''
                    SELECT file_path, file_type, analysis_result
                    FROM electrical_uploaded_files
                    WHERE file_name = ? AND session_id = ?
                ''', (file_ref, session_id))
                file_info = cursor.fetchone()
                if file_info:
                    file_path, file_type, analysis_result = file_info
                    user_message += f"\n[Analysis of {file_ref}: {analysis_result}]"
            conn.close()

        if electrical_qa_chain:
            result = electrical_qa_chain.invoke({"query": user_message})
            raw_bot_response = validate_response(result)
            # Convert markdown to HTML with proper Flask Markup
            html_bot_response = md_to_html(raw_bot_response)
        else:
            html_bot_response = Markup("I'm having trouble accessing the electrical knowledge base. Please try again.")

        response_time = (datetime.now() - start_time).total_seconds()
        store_conversation(session_id, user_message, raw_bot_response, response_time, 'electrical')
        
        log_chat_interaction(
            session.get('user_id'),
            session.get('username'),  
            user_message,
            raw_bot_response,
            session_id,
            'electrical'
        )

        return jsonify({
            "bot_response": str(html_bot_response),  # Convert Markup to string for JSON
            "session_id": session_id,
            "response_time": response_time,
            "citations": []
        })

    except Exception as e:
        error_response = f"Error processing electrical specification request: {str(e)}"
        store_conversation(session_id, user_message, error_response, 0, 'electrical')
        return jsonify({"bot_response": error_response})

@app.route("/history/<session_id>", methods=["GET"])
@login_required
def get_history(session_id):
    history = get_conversation_history(session_id, 10, 'material')
    return jsonify({"history": history})

@app.route("/electrical/history/<session_id>", methods=["GET"])
@login_required
def get_electrical_history(session_id):
    history = get_conversation_history(session_id, 10, 'electrical')
    return jsonify({"history": history})

@app.route("/api/stats", methods=["GET"])
@login_required
def get_stats():
    """Enhanced stats endpoint for material workspace"""
    try:
        conn = sqlite3.connect('material_specification_chat.db')
        cursor = conn.cursor()

        # Get material stats
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM user_sessions")
        total_sessions = cursor.fetchone()[0]
        
        total_files = get_uploaded_files_count('material')
        
        cursor.execute("SELECT AVG(response_time) FROM conversations WHERE response_time > 0")
        avg_result = cursor.fetchone()
        avg_response_time = round(float(avg_result[0]), 2) if avg_result[0] else 0.0

        conn.close()

        response_data = {
            "total_conversations": total_conversations,
            "total_sessions": total_sessions,
            "total_materials": 0,  # Kept for compatibility
            "total_files": total_files,
            "avg_response_time": avg_response_time,
            "timestamp": datetime.now().isoformat(),
            "user": session.get('username', 'Unknown')
        }

        return jsonify(response_data)

    except Exception as e:
        error_response = {
            "total_conversations": 0,
            "total_sessions": 0,
            "total_materials": 0,
            "total_files": 0,
            "avg_response_time": 0.0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(error_response), 500

@app.route("/api/electrical/stats", methods=["GET"])
@login_required
def get_electrical_stats():
    """Enhanced stats endpoint for electrical workspace"""
    try:
        conn = sqlite3.connect('material_specification_chat.db')
        cursor = conn.cursor()

        # Get electrical stats
        cursor.execute("SELECT COUNT(*) FROM electrical_conversations")
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM electrical_user_sessions")
        total_sessions = cursor.fetchone()[0]
        
        total_files = get_uploaded_files_count('electrical')
        
        cursor.execute("SELECT AVG(response_time) FROM electrical_conversations WHERE response_time > 0")
        avg_result = cursor.fetchone()
        avg_response_time = round(float(avg_result[0]), 2) if avg_result[0] else 0.0

        conn.close()

        response_data = {
            "total_conversations": total_conversations,
            "total_sessions": total_sessions,
            "total_materials": 0,  # Not applicable for electrical
            "total_files": total_files,
            "avg_response_time": avg_response_time,
            "timestamp": datetime.now().isoformat(),
            "user": session.get('username', 'Unknown')
        }

        return jsonify(response_data)

    except Exception as e:
        error_response = {
            "total_conversations": 0,
            "total_sessions": 0,
            "total_materials": 0,
            "total_files": 0,
            "avg_response_time": 0.0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(error_response), 500

if __name__ == "__main__":
    print("🔧 Enhanced Material & Electrical Specification Assistant Backend Starting...")
    print("🔐 Authentication system active with improved CSV handling")
    print("📊 Database initialized for material and electrical tracking")
    print("🤖 LangChain + Groq + Chroma pipeline ready for specifications")
    print("⚡ Electrical workspace enabled with separate data handling")
    print("📋 Professional table rendering enabled with enhanced styling")
    print("🌐 Server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)