# (🏠 PropertyPal_bot) Real Estate Agent - Zameen.com Dataset Assistant

An intelligent real estate chatbot powered by OpenAI GPT-4 and LangChain that helps users find properties from the Zameen.com dataset using advanced Retrieval-Augmented Generation (RAG) technology.

## 📋 Project Overview

This Real Estate Agent is built specifically for the Pakistani real estate market using Zameen.com property data. It provides intelligent property search capabilities through a conversational interface, helping users find their ideal homes, apartments, and commercial properties across major Pakistani cities.

## ✨ Features

- **🤖 Smart Query Classification**: Automatically distinguishes between property searches and general conversation
- **🔍 RAG-Powered Search**: Advanced vector similarity search for accurate property matching
- **💬 Conversational Interface**: Natural language interaction built with Streamlit
- **📊 Zameen.com Integration**: Real property data from Pakistan's leading real estate platform
- **🏪 Persistent Storage**: ChromaDB vector database for fast retrieval
- **🌍 Multi-City Support**: Properties across Karachi, Lahore, Islamabad, and other major cities
- **💰 Price & Filter Options**: Search by budget, bedrooms, area, and location

## 🛠️ Prerequisites

Before setting up the project, ensure you have:

- **Python 3.8+** installed on your system
- **OpenAI API Key** from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Git** for cloning the repository
- **Internet connection** for initial setup and API calls

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Muhammad-Yousaf09/PropertyPal_bot.git

# Navigate to project directory
cd real-estate-agent-zameen
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Setup Dataset

Ensure you have the `Property_data.csv` file in your project root directory. This file should contain Zameen.com property data with the following structure:

```csv
Location,Price,Area,Bedrooms,Bathrooms,Date Added,Agency,Agent,Page URL,Property Type
DHA Phase 5 Karachi,15000000,1800,4,3,2024-01-15,ABC Realty,Ahmed Ali,https://zameen.com/123,House
Gulberg Lahore,8500000,1200,3,2,2024-01-10,XYZ Properties,Sara Khan,https://zameen.com/456,Apartment
F-8 Islamabad,25000000,2500,5,4,2024-01-08,Prime Estate,Hassan Ahmed,https://zameen.com/789,House
```

## 🏃‍♂️ Running the Application

### Local Development

1. **Start the Streamlit server:**
   ```bash
   streamlit run app.py
   ```

2. **Access the application:**
   - Open your browser and navigate to: `http://localhost:8501`
   - The application will automatically open in your default browser

3. **Configure the application:**
   - Enter your OpenAI API Key in the sidebar
   - Wait for the system to initialize (first run may take 1-2 minutes)

4. **Start exploring properties:**
   - Use the chat interface to search for properties
   - Try example queries or ask in natural language

## 💡 Usage Examples

### Property Search Queries
```
🏠 "Show me 3 bedroom houses in DHA Karachi under 2 crore"
🏢 "Find apartments in Gulberg Lahore with 2+ bathrooms"
💰 "Properties under 50 lakh in Islamabad"
📍 "Houses for sale in Clifton with sea view"
🛏️ "4 bedroom houses in Model Town Lahore"
```

### General Conversation
```
👋 "Hello, how are you?"
❓ "What areas do you cover?"
💬 "Thank you for your help"
```

## 📁 Project Structure

```
real-estate-agent-zameen/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── Property_data.csv        # Zameen.com dataset (required)
├── chroma_db/              # Vector database (auto-created)
│   ├── chroma.sqlite3      # ChromaDB storage
├── .gitignore              # Git ignore file
└── screenshots/            # Application screenshots (optional)
```

## ⚙️ Configuration Options

### Environment Variables
Set your OpenAI API key as an environment variable (optional):

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# macOS/Linux
export OPENAI_API_KEY="your-api-key-here"
```

### Customizable Parameters

In `app.py`, you can modify:

```python
# Text splitting configuration
chunk_size=1000,           # Size of text chunks
chunk_overlap=100,         # Overlap between chunks

# Retrieval configuration
search_kwargs={"k": 5}     # Number of similar documents to retrieve

# Model configuration
model="gpt-4o",           # OpenAI model to use
temperature=0,            # Response randomness (0-1)
```

## 🔧 Technical Architecture

### RAG Pipeline
1. **Data Loading**: CSV property data is loaded using LangChain CSVLoader
2. **Text Chunking**: Documents split into manageable chunks with overlap
3. **Embeddings**: OpenAI text-embedding-3-large creates vector representations
4. **Vector Storage**: ChromaDB stores embeddings for fast similarity search
5. **Query Classification**: GPT-4 determines if queries are property-related
6. **Retrieval**: Relevant property documents retrieved based on similarity
7. **Generation**: GPT-4 generates contextual responses using retrieved data

### Models Used
- **Embeddings**: `text-embedding-3-large` (~$0.00013 per 1K tokens)
- **Chat**: `gpt-4o` (~$0.005 input, ~$0.015 output per 1K tokens)

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### ❌ "Property_data.csv file not found!"
```bash
# Ensure the CSV file is in the project root
ls -la Property_data.csv  # Linux/macOS
dir Property_data.csv     # Windows
```

#### ❌ "Invalid API Key"
- Verify your OpenAI API key is correct
- Check API credit balance at OpenAI Platform
- Ensure no extra spaces in the API key

#### ⏳ Slow Initialization
- First run processes the entire dataset (normal)
- Subsequent runs use cached embeddings (faster)
- Large datasets may take 2-3 minutes initially

#### 💾 Memory Issues
Reduce chunk size for large datasets:
```python
chunk_size=500,    # Reduce from 1000
chunk_overlap=50,  # Reduce from 100
```

#### 🌐 API Rate Limits
- OpenAI has rate limits for API calls
- Wait a few minutes if you hit limits
- Consider upgrading your OpenAI plan for higher limits

## 📊 Dataset Information

### Zameen.com Data Structure
The project expects property data with these columns:
- **Location**: Property address/area
- **Price**: Property price in PKR
- **Area**: Property size in sq ft/marla
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms  
- **Date Added**: Listing date
- **Agency**: Real estate agency name
- **Agent**: Agent contact information
- **Page URL**: Original Zameen.com listing
- **Property Type**: House/Apartment/Plot etc.

### Supported Cities
- Karachi (DHA, Clifton, Gulshan, etc.)
- Lahore (DHA, Gulberg, Model Town, etc.)
- Islamabad (F-sectors, G-sectors, etc.)
- Rawalpindi

## 💰 Cost Estimation

### API Usage Costs (Approximate)
- **Setup**: $0.10-0.50 (one-time embedding creation)
- **Per Query**: $0.01-0.05 (depending on complexity)
- **Monthly Usage**: $5-20 for moderate use (100-500 queries)

## 🚀 Deployment Options

### Streamlit Cloud (Free)
```bash
# Push to GitHub
git add .
git commit -m "Initial commit"
git push origin main

# Deploy on Streamlit Cloud
# Visit: https://share.streamlit.io
# Connect your GitHub repository
```

### Heroku Deployment
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT --server.enableCORS false" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** and test thoroughly
4. **Commit changes**: `git commit -m "Add new feature"`
5. **Push to branch**: `git push origin feature/new-feature`
6. **Submit a Pull Request**

### Development Guidelines
- Follow PEP 8 Python style guide
- Add comments for complex functions
- Test with different property queries
- Update documentation for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Zameen.com** for providing comprehensive Pakistani real estate data
- **OpenAI** for powerful language models and embeddings
- **LangChain** for RAG framework and document processing
- **Streamlit** for the intuitive web interface
- **ChromaDB** for efficient vector storage

## 📧 Support & Contact

If you encounter any issues or have questions:

1. **Check the troubleshooting section** above
2. **Review your API key** and dataset format
3. **Open an issue** on GitHub with detailed error messages
4. **Contact**: your-email@example.com

## 🔮 Future Enhancements

- [ ] Multi-language support (Urdu/English)
- [ ] Property image analysis
- [ ] Price prediction features
- [ ] WhatsApp integration
- [ ] Mobile app version
- [ ] Advanced filtering options
- [ ] Property comparison features

---

**🏡 Happy house hunting with your AI Real Estate Agent! 🤖**

*Built with ❤️ for the Pakistani real estate market*
