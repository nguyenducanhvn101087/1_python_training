# Copilot Instructions for Python Training Projects

## Project Architecture

This is a Python training repository with 4 distinct learning modules, each demonstrating different AI/ML concepts:

- **Basic_python/**: Foundation Python concepts (simple scripts)
- **Basic_Image_Classifier/**: Computer vision with TensorFlow/Keras + Streamlit UI
- **Simple_AI_Agent/**: LangChain/LangGraph conversational agent with tool usage
- **CV_Critiquer/**: Document processing + OpenAI API integration with Streamlit

Each project is self-contained with its own `pyproject.toml` and dependencies.

## Essential Development Workflow

### Dependency Management
- **All projects use `uv` for dependency management** - never use `pip` directly
- Run applications: `uv run streamlit run main.py` (for Streamlit apps) or `uv run main.py` (for CLI)
- Install packages: `uv add package-name` (automatically updates pyproject.toml)
- The `uv.lock` files are committed and should be preserved

### Project Structure Patterns
```
ProjectName/
├── main.py          # Single entry point (no module splitting)
├── pyproject.toml   # uv-managed dependencies
├── README.md        # Run instructions only
└── uv.lock         # Locked dependencies
```

### Code Conventions

**Streamlit Apps** (Basic_Image_Classifier, CV_Critiquer):
- Use `@st.cache_resource` for expensive operations (model loading)
- `st.set_page_config()` with custom titles and centered layout
- Error handling with `st.error()` and `st.spinner()` for UX

**AI Integration Patterns**:
- Environment variables in `.env` files for API keys (`OPENAI_API_KEY`)
- Direct OpenAI client usage: `OpenAI(api_key=OPENAI_API_KEY)`
- LangChain agents: Use `create_react_agent()` with `@tool` decorated functions

**File Processing**:
- PyPDF2 for PDF text extraction with `io.BytesIO()` for Streamlit uploads
- Always check for empty content: `if not content.strip():`

## Project-Specific Requirements

### Python Versions
- Basic_Image_Classifier & CV_Critiquer: `>=3.11`
- Simple_AI_Agent: `>=3.10`
- Basic_python: No specific requirement

### Key Dependencies by Project
- **Image Classifier**: `tensorflow>=2.20.0`, `opencv-python`, `streamlit`
- **AI Agent**: `langchain`, `langgraph`, `torch`, `transformers`
- **CV Critiquer**: `openai>=2.1.0`, `pypdf2`, `streamlit`

## Critical Implementation Details

### Model Loading Pattern
```python
@st.cache_resource
def load_cached_model():
    return MobileNetV2(weights='imagenet')
```

### LangChain Tool Definition
```python
@tool
def tool_name(param: type) -> str:
    """Clear docstring for agent decision-making."""
    return result
```

### File Upload Handling
```python
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")
```

## Development Notes

- All `main.py` files are standalone - no module imports between projects
- Extensive inline comments explain ML concepts for learning purposes
- TODO comments indicate planned improvements in each project
- Cross-platform considerations: Linux primary, Windows Git Bash secondary