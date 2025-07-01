#!/bin/bash

# Material Specification Identification Chatbot Setup Script
# This script sets up the complete project structure and dependencies

echo "🏭 Setting up Material Specification Identification Chatbot..."
echo "=================================================="

# Create project directory structure
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p vectordb
mkdir -p templates
mkdir -p logs

# Move HTML template to correct location
if [ -f "material_index.html" ]; then
    mv material_index.html templates/
    echo "✅ Moved HTML template to templates directory"
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv material_chatbot_env

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source material_chatbot_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file from template
if [ ! -f ".env" ]; then
    cp env_template.env .env
    echo "📝 Created .env file from template"
    echo "⚠️  Please edit .env file and add your GROQ_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Create sample data files
echo "📄 Creating sample data files..."

# Sample material specifications CSV
cat > data/material_specifications.csv << 'EOL'
Material_Code,Description,Category,Specification,WO_Number,MR_Number,PR_Number,RFQ_Number,Status
MAT001,Steel Pipe ASTM A106,Raw Material,Grade B Carbon Steel Seamless Pipe,WO-2024-001,MR-001-2024,PR-2024-001,RFQ-001-2024,Active
MAT002,Stainless Steel 316L,Raw Material,Austenitic Stainless Steel,WO-2024-002,MR-002-2024,PR-2024-002,RFQ-002-2024,Pending
MAT003,Carbon Steel Plate,Raw Material,ASTM A36 Structural Steel,WO-2024-003,MR-003-2024,PR-2024-003,RFQ-003-2024,Delivered
MAT004,Aluminum Alloy 6061,Raw Material,T6 Temper Aluminum,WO-2024-004,MR-004-2024,PR-2024-004,RFQ-004-2024,In Transit
MAT005,Copper Pipe Type K,Raw Material,ASTM B88 Hard Drawn,WO-2024-005,MR-005-2024,PR-2024-005,RFQ-005-2024,Inspection
EOL

# Sample material standards text file
cat > data/material_standards.txt << 'EOL'
MATERIAL SPECIFICATION STANDARDS

1. STEEL MATERIALS
- ASTM A106: Standard Specification for Seamless Carbon Steel Pipe for High-Temperature Service
- ASTM A36: Standard Specification for Carbon Structural Steel
- ASTM A516: Standard Specification for Pressure Vessel Plates, Carbon Steel

2. STAINLESS STEEL MATERIALS
- ASTM A312: Standard Specification for Seamless, Welded, and Heavily Cold Worked Austenitic Stainless Steel Pipes
- Grade 316L: Low carbon austenitic stainless steel with excellent corrosion resistance

3. ALUMINUM MATERIALS
- ASTM B221: Standard Specification for Aluminum and Aluminum-Alloy Extruded Bars, Rods, Wire, Profiles, and Tubes
- 6061-T6: Heat-treated aluminum alloy with good mechanical properties

4. COPPER MATERIALS
- ASTM B88: Standard Specification for Seamless Copper Water Tube
- Type K: Heavy wall copper tubing for underground service

5. QUALITY CONTROL REQUIREMENTS
- All materials must undergo PMI (Positive Material Identification) testing
- Material certificates must be provided for all critical materials
- Inspection reports required for pressure-containing components

6. PROCUREMENT WORKFLOW
- Work Order (WO) generation for material requirements
- Material Request (MR) creation with line items and dates
- Purchase Request (PR) processing with buyer assignment
- Request for Quotation (RFQ) with supplier evaluation
- Purchase Order (PO) with delivery schedules
- Goods receipt and inspection procedures
- Final delivery to site with status tracking
EOL

# Create sample procurement procedures PDF content
cat > data/procurement_procedures.txt << 'EOL'
PROCUREMENT PROCEDURES FOR MATERIAL SPECIFICATION IDENTIFICATION

OVERVIEW:
This document outlines the standard procedures for material procurement, specification identification, and tracking within our organization.

WORKFLOW STAGES:

1. WORK ORDER (WO) CREATION
- Engineering creates work orders specifying material requirements
- Each WO contains detailed material specifications and quantities
- WO numbers follow format: WO-YYYY-### (e.g., WO-2024-001)

2. MATERIAL REQUEST (MR) PROCESSING
- Materials department creates MR based on WO requirements
- MR includes line items with specific material codes
- MR dates track when requests are initiated
- Format: MR-###-YYYY (e.g., MR-001-2024)

3. PURCHASE REQUEST (PR) GENERATION
- Procurement team converts approved MRs into PRs
- PR line items match MR specifications
- PR created dates logged for tracking
- Buyers assigned based on material category and value
- Format: PR-YYYY-### (e.g., PR-2024-001)

4. REQUEST FOR QUOTATION (RFQ)
- RFQs sent to qualified suppliers
- RFQ line items include technical specifications
- RFQ raised dates tracked for supplier response times
- Evaluation criteria includes price, delivery, and quality
- Format: RFQ-###-YYYY (e.g., RFQ-001-2024)

5. PURCHASE ORDER (PO) PROCESSING
- POs issued to selected suppliers
- PO delivery dates established
- Terms and conditions include material specifications
- Expediting procedures for critical materials

6. DELIVERY TRACKING
- Actual delivery status monitored
- Store inspection status recorded
- Quality control checks performed
- Site delivery status updated
- Non-conformance procedures for rejected materials

MATERIAL IDENTIFICATION REQUIREMENTS:
- All materials must have positive material identification (PMI)
- Material certificates required for critical components
- Traceability maintained from procurement to installation
- Non-destructive testing where specified

DOCUMENTATION REQUIREMENTS:
- Material test certificates (MTCs)
- Mill test certificates for steel products
- Chemical composition reports
- Mechanical property test results
- Dimensional inspection reports

APPROVAL AUTHORITIES:
- Engineering: Technical specification approval
- Materials: MR approval and material substitutions
- Procurement: PR approval and supplier selection
- Quality: Inspection and acceptance authority
- Project Management: Overall coordination and expediting

SYSTEM INTEGRATION:
- ERP system integration for real-time status updates
- Document management for certificate storage
- Supplier portal for delivery status updates
- Mobile applications for field inspection

PERFORMANCE METRICS:
- On-time delivery performance
- Material quality acceptance rates
- Procurement cycle time
- Cost savings through supplier negotiations
- Inventory turnover ratios
EOL

echo "✅ Sample data files created in data/ directory"

# Create startup script
cat > start_chatbot.sh << 'EOL'
#!/bin/bash
echo "🏭 Starting Material Specification Identification Chatbot..."
source material_chatbot_env/bin/activate
python material_chatbot_app.py
EOL
chmod +x start_chatbot.sh

# Create project structure summary
cat > PROJECT_STRUCTURE.md << 'EOL'
# Material Specification Identification Chatbot - Project Structure

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
├── .env                         # Environment variables (create from template)
├── env_template.env            # Environment template
├── setup.sh                    # This setup script
├── start_chatbot.sh            # Startup script
└── PROJECT_STRUCTURE.md        # This file
```

## Quick Start
1. Edit .env file with your GROQ_API_KEY
2. Run: ./start_chatbot.sh
3. Open: http://localhost:5000

## Features
- Material specification identification using AI
- Work Order (WO), Material Request (MR), Purchase Request (PR), RFQ tracking
- Industrial-themed responsive UI
- SQLite database for conversation and material tracking
- Document processing from data/ folder
- Real-time chat with material management expertise
EOL

echo ""
echo "🎉 Setup Complete!"
echo "=================================================="
echo "Next steps:"
echo "1. Edit .env file and add your GROQ_API_KEY"
echo "2. Add your material specification documents to data/ folder"
echo "3. Run: ./start_chatbot.sh"
echo "4. Open http://localhost:5000 in your browser"
echo ""
echo "📂 Project structure created in current directory"
echo "📊 Sample data files added to data/ directory"
echo "🔧 Ready to start your Material Specification Identification Chatbot!"