# Material Specification Identification Chatbot

A comprehensive AI-powered chatbot system for material specification identification, supply chain management, and procurement tracking. Built with Flask, LangChain, and Groq API following the same architecture as your mental health chatbot.

## Features

### 🏭 Material Management
- **Material Specification Identification**: AI-powered material classification and identification
- **Work Order (WO) Tracking**: Complete work order lifecycle management
- **Material Request (MR) Processing**: Line item tracking with dates and statuses
- **Purchase Request (PR) Workflows**: Buyer assignment and approval tracking
- **Request for Quotation (RFQ)**: Supplier quotation management
- **Delivery Status Tracking**: Real-time delivery and inspection status

### 🤖 AI Capabilities
- **Document Processing**: Automatically processes PDFs, CSVs, and text files from data folder
- **Intelligent Responses**: Specialized in material standards, procurement procedures, and supply chain
- **Context-Aware Chat**: Maintains conversation history and provides relevant responses
- **Quick Actions**: Pre-defined queries for common material management tasks

### 💾 Data Management
- **SQLite Database**: Stores conversation history and material tracking records
- **Session Tracking**: User session management with statistics
- **Real-time Analytics**: Live dashboard with system metrics
- **Document Storage**: Vector database for efficient document retrieval

### 🎨 Industrial UI Design
- **Professional Theme**: Industrial blue color scheme optimized for manufacturing environments
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Status**: Connection status and typing indicators
- **Dashboard Sidebar**: Quick stats and action buttons

## Quick Start

### Prerequisites
- Python 3.8+
- Groq API key

### Installation

1. **Clone and setup:**
   ```bash
   # Make setup script executable
   chmod +x setup.sh
   
   # Run setup
   ./setup.sh
   ```

2. **Configure API key:**
   ```bash
   # Edit .env file
   nano .env
   
   # Add your Groq API key
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

3. **Add your data:**
   ```bash
   # Add your material specification documents to data folder
   cp your_material_docs.pdf data/
   cp your_specifications.csv data/
   ```

4. **Start the chatbot:**
   ```bash
   ./start_chatbot.sh
   ```

5. **Access the interface:**
   Open http://localhost:5000 in your browser

## Project Structure

```
material_chatbot_project/
├── material_chatbot_app.py      # Main Flask application
├── templates/
│   └── material_index.html      # Frontend HTML template
├── data/                        # Document storage for RAG
│   ├── material_specifications.csv
│   ├── material_standards.txt
│   └── procurement_procedures.txt
├── vectordb/                    # ChromaDB vector storage
├── logs/                        # Application logs
├── material_chatbot_env/        # Python virtual environment
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── setup.sh                     # Setup script
├── start_chatbot.sh            # Startup script
└── README.md                   # This file
```

## Usage Examples

### Material Identification Queries
- "What is the specification for ASTM A106 steel pipe?"
- "Show me materials with WO-2024-001"
- "List all pending material requests"

### Procurement Tracking
- "What's the status of PR-2024-002?"
- "Show RFQ timeline for stainless steel materials"
- "Which materials are delayed in delivery?"

### Quality Control
- "PMI testing requirements for carbon steel"
- "Material certificate requirements"
- "Inspection procedures for aluminum alloys"

## Technical Architecture

### Backend (Flask + LangChain)
- **Flask**: Web framework with CORS support
- **LangChain**: Document processing and retrieval
- **ChromaDB**: Vector database for embeddings
- **Groq API**: Language model (llama3-70b-8192)
- **SQLite**: Local database for tracking

### Frontend (HTML/CSS/JS)
- **Responsive Design**: Grid layout with sidebar dashboard
- **Industrial Theme**: Blue/gray color scheme
- **Real-time Updates**: AJAX for seamless chat experience
- **Progressive Enhancement**: Works without JavaScript

### Document Processing
- **Multi-format Support**: PDF, CSV, TXT files
- **Chunking Strategy**: 500 characters with 50 overlap
- **Embeddings**: HuggingFace sentence-transformers
- **Retrieval**: Top-3 relevant chunks for context

## Database Schema

### Conversations Table
- `id`: Primary key
- `user_id`: Session identifier
- `user_message`: User input
- `bot_response`: AI response
- `timestamp`: Message time
- `response_time`: Processing duration

### Material Records Table
- `wo_number`: Work Order number
- `mr_number`: Material Request number
- `pr_number`: Purchase Request number
- `rfq_number`: Request for Quotation number
- `buyer_assigned`: Assigned buyer
- `delivery_status`: Current status
- `inspection_status`: Quality control status

## API Endpoints

### Chat API
```
POST /api/chat
Content-Type: application/json

{
  "message": "Your material specification query"
}
```

### Material Records API
```
POST /api/material-record
Content-Type: application/json

{
  "wo_number": "WO-2024-001",
  "mr_number": "MR-001-2024",
  "status": "Active"
}
```

### Statistics API
```
GET /api/stats

Returns:
{
  "total_conversations": 150,
  "avg_response_time": 1.2,
  "total_materials": 45
}
```

## Customization

### Adding New Material Types
1. Update the prompt template in `material_chatbot_app.py`
2. Add new quick action buttons in the HTML template
3. Create specific CSV templates for new material categories

### Extending Database Schema
1. Modify the `init_database()` function
2. Add new API endpoints for additional data types
3. Update the frontend dashboard accordingly

### Styling Modifications
The industrial color scheme uses:
- Primary: `#05aff6` (bright blue)
- Secondary: `#0c2c34` (dark blue-gray)
- Accent: `#90defc` (light blue)
- Background: `#1e303c` (medium blue-gray)

## Troubleshooting

### Common Issues

1. **"TemplateNotFound" Error**
   ```bash
   # Ensure HTML file is in templates directory
   ls templates/material_index.html
   ```

2. **"Invalid API Key" Error**
   ```bash
   # Check .env file
   cat .env | grep GROQ_API_KEY
   ```

3. **No Documents Found**
   ```bash
   # Add documents to data folder
   ls data/
   ```

4. **Port Already in Use**
   ```bash
   # Check if port 5000 is available
   lsof -i :5000
   ```

### Performance Optimization
- Place frequently accessed documents in data/
- Use descriptive filenames for better retrieval
- Regularly clean up old conversation logs
- Monitor vector database size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and follows the same license terms as your mental health chatbot project.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the project structure
3. Ensure all prerequisites are met
4. Contact the development team

---

**Ready to revolutionize your material specification identification process!** 🏭🚀