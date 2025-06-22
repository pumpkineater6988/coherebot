# 🚀 Cohere AI Chatbot

A high-performance chatbot powered by Cohere AI with document processing capabilities and optimized for real-time conversations without lag.

## ✨ Features

- **🤖 Cohere AI Integration**: Powered by Cohere for intelligent responses
- **📁 Document Processing**: Upload PDF, CSV, and Excel files for context-aware responses
- **⚡ Performance Optimized**: No message lagging with connection pooling and caching
- **🎨 Modern UI**: Beautiful gradient design with smooth animations
- **📊 Real-time Status**: Live monitoring of system status and metrics
- **📤 Export Functionality**: Download chat history as CSV
- **🔄 Session Management**: Persistent chat history and knowledge base

## 🛠️ Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `env.example` to `.env`
   - Add your Cohere API key:
   ```
   COHERE_API_KEY=your_actual_cohere_api_key_here
   COHERE_API_BASE_URL=https://api.cohere.ai/v1
   COHERE_MODEL=command-r-plus
   ```

## 🚀 Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the provided URL

3. **Optional**: Upload documents (PDF, CSV, Excel) to enhance responses

4. **Start chatting** with Cohere AI!

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COHERE_API_KEY` | Your Cohere API key | Required |
| `COHERE_API_BASE_URL` | Cohere API endpoint | `https://api.cohere.ai/v1` |
| `COHERE_MODEL` | Model to use | `command-r-plus` |
| `MAX_TOKENS` | Maximum response length | `4096` |
| `TEMPERATURE` | Response creativity | `0.7` |

### Performance Settings

The app includes several performance optimizations:

- **Connection Pooling**: Reuses HTTP connections for faster API calls
- **Caching**: Embeds models and file processing results
- **Async Processing**: Non-blocking file uploads and processing
- **Memory Management**: Efficient handling of large documents

## 📁 Project Structure

```
Cohere App/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
├── env.example        # Environment variables template
└── README.md          # This file
```

## 🔧 Performance Optimizations

### 1. **Connection Pooling**
- Reuses HTTP connections to reduce latency
- Configurable pool size and retry logic

### 2. **Caching Strategy**
- Embeds model caching with `@st.cache_resource`
- File processing results cached with `@st.cache_data`

### 3. **Memory Management**
- Efficient document chunking and processing
- Automatic cleanup of temporary files

### 4. **UI Optimizations**
- Smooth animations and transitions
- Responsive design with proper loading states
- Real-time status updates

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Cohere API key is correctly set in `.env`
   - Verify the API key has proper permissions

2. **File Upload Issues**
   - Check file format (PDF, CSV, Excel only)
   - Ensure file size is reasonable (< 50MB recommended)

3. **Performance Issues**
   - Close other applications to free up memory
   - Check internet connection for API calls
   - Restart the application if needed

### Error Messages

- **"Cohere API request failed"**: Check API key and network connection
- **"Error processing file"**: Verify file format and try smaller files
- **"Export error"**: Ensure chat history exists before exporting

## 🔒 Security

- API keys are stored in environment variables
- No sensitive data is logged or stored
- Temporary files are automatically cleaned up
- HTTPS connections for all API calls

## 📈 Performance Metrics

The app is optimized for:
- **Response Time**: < 2 seconds for typical queries
- **Memory Usage**: Efficient handling of large documents
- **Concurrent Users**: Supports multiple simultaneous users
- **File Processing**: Fast document embedding and indexing

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational and personal use.

---

**🚀 Enjoy chatting with Cohere AI!** 