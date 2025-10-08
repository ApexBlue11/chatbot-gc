import streamlit as st
import requests
import pandas as pd
import json
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Genetic Variant Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.transcript-box {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
}
.prediction-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.stChatMessage {
    background-color: 36454F;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== QUERY CLASSIFICATION ====================

@dataclass
class QueryClassification:
    is_genomic: bool
    query_type: str
    extracted_identifier: Optional[str]

class GenomicQueryRouter:
    def __init__(self):
        self.hgvs_patterns = {
            'transcript': [
                r'\b(NM_\d+(?:\.\d+)?):c\.[A-Za-z0-9\-+*>_]+',
                r'\b(ENST\d+(?:\.\d+)?):c\.[A-Za-z0-9\-+*>_]+',
            ],
            'genomic': [
                r'\b(NC_\d+(?:\.\d+)?):g\.[A-Za-z0-9\-+*>_]+',
                r'\b(chr(?:\d+|X|Y|MT?)):g\.\d+[A-Za-z]+>[A-Za-z]+',
            ],
            'protein': [
                r'\b(NP_\d+(?:\.\d+)?):p\.[A-Za-z0-9\-+*>_()]+',
                r'\b(ENSP\d+(?:\.\d+)?):p\.[A-Za-z0-9\-+*>_()]+',
            ]
        }
        self.rsid_pattern = r'\b(rs\d+)\b'
    
    def classify_query(self, query: str) -> QueryClassification:
        query = query.strip()
        
        for variant_type, patterns in self.hgvs_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    return QueryClassification(
                        is_genomic=True,
                        query_type=f'hgvs_{variant_type}',
                        extracted_identifier=match.group(0)
                    )
        
        rsid_match = re.search(self.rsid_pattern, query, re.IGNORECASE)
        if rsid_match:
            return QueryClassification(
                is_genomic=True,
                query_type='rsid',
                extracted_identifier=rsid_match.group(1)
            )
        
        return QueryClassification(
            is_genomic=False,
            query_type='general',
            extracted_identifier=None
        )

# ==================== DEBUG UTILITIES ====================

def log_debug(message: str, data: any = None, level: str = "INFO"):
    """Comprehensive debug logging with timestamps."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message,
        'data': data
    }
    
    st.session_state.debug_logs.append(log_entry)
    
    # Also print to console for server-side debugging
    print(f"[{timestamp}] [{level}] {message}")
    if data:
        print(f"  Data: {json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)}")


def show_debug_panel():
    """Display debug logs in an expandable section."""
    if 'debug_logs' not in st.session_state or not st.session_state.debug_logs:
        return
    
    with st.expander("🐛 Debug Logs (Click to expand)", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Logs", key="clear_debug_logs"):
                st.session_state.debug_logs = []
                st.rerun()
        
        # Reverse order to show latest first
        for log in reversed(st.session_state.debug_logs[-50:]):  # Show last 50 logs
            level_emoji = {
                'INFO': 'ℹ️',
                'SUCCESS': '✅',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'DEBUG': '🔍'
            }.get(log['level'], '📝')
            
            st.markdown(f"**{level_emoji} [{log['timestamp']}] {log['message']}**")
            
            if log.get('data'):
                with st.container():
                    st.json(log['data'], expanded=False)

# ==================== API KEY MANAGEMENT ====================

def get_manual_api_key(service: str) -> Optional[str]:
    """Gets API key with proper state management."""
    key_name = f"{service.lower()}_api_key"
    
    log_debug(f"Checking for {service} API key", level="DEBUG")
    
    # Check if already in session state
    if key_name in st.session_state and st.session_state[key_name]:
        log_debug(f"{service} API key found in session", level="SUCCESS")
        return st.session_state[key_name]
    
    # Request from user
    st.warning(f"{service} API key not found. Please enter it below.")
    manual_key = st.text_input(
        f"Enter your {service} API Key:",
        type="password",
        key=f"{key_name}_input",
        help=f"Your API key will be stored only for this session"
    )
    
    if manual_key:
        st.session_state[key_name] = manual_key
        log_debug(f"{service} API key saved to session", level="SUCCESS")
        st.success(f"{service} API key saved! Reloading...")
        time.sleep(0.5)
        st.rerun()
    
    return None

# ==================== OPENAI API ====================

def call_openai_api(prompt: str, api_key: str, context: List[Dict[str, str]]) -> str:
    """Calls OpenAI Chat Completions API with comprehensive error handling and debugging."""
    api_url = "https://api.openai.com/v1/chat/completions"
    
    log_debug("Starting OpenAI API call", level="INFO")
    
    # Build headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Build messages array - OpenAI expects [{role, content}, ...]
    messages = context.copy()
    messages.append({"role": "user", "content": prompt})
    
    # Build payload
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    log_debug("OpenAI Request Payload", {
        "url": api_url,
        "model": payload["model"],
        "message_count": len(messages),
        "last_message_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "full_payload": payload
    }, level="DEBUG")
    
    # Store for UI display
    st.session_state['last_ai_payload'] = payload
    st.session_state['last_ai_service'] = 'OpenAI'
    
    try:
        with st.spinner("🤖 OpenAI is thinking..."):
            start_time = time.time()
            
            log_debug("Sending request to OpenAI...", level="INFO")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            log_debug(f"OpenAI response received in {elapsed:.2f}s", level="SUCCESS")
            
            # Log response details
            log_debug("OpenAI Response Details", {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "elapsed_seconds": elapsed
            }, level="DEBUG")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            log_debug("OpenAI Response Data", response_data, level="DEBUG")
            
            # Extract content
            if "choices" not in response_data or not response_data["choices"]:
                log_debug("No choices in response", response_data, level="ERROR")
                return "OpenAI returned no response choices."
            
            content = response_data["choices"][0]["message"]["content"].strip()
            
            # Log token usage
            if "usage" in response_data:
                usage = response_data["usage"]
                log_debug(f"Token usage: {usage.get('total_tokens', 'N/A')} total "
                         f"({usage.get('prompt_tokens', 'N/A')} prompt + "
                         f"{usage.get('completion_tokens', 'N/A')} completion)", 
                         usage, level="INFO")
            
            log_debug(f"OpenAI success! Response length: {len(content)} chars", level="SUCCESS")
            
            return content
            
    except requests.exceptions.Timeout:
        error_msg = "OpenAI request timed out after 60 seconds"
        log_debug(error_msg, level="ERROR")
        st.error(f"⏱️ {error_msg}")
        return f"Request timed out. The API might be slow or the prompt too complex."
        
    except requests.exceptions.HTTPError as http_err:
        log_debug(f"OpenAI HTTP error: {http_err}", level="ERROR")
        
        try:
            err_json = http_err.response.json()
            error_details = err_json.get("error", {})
            error_type = error_details.get("type", "unknown")
            error_code = error_details.get("code", "unknown")
            error_msg = error_details.get("message", str(http_err))
            
            log_debug("OpenAI Error Details", {
                "status_code": http_err.response.status_code,
                "error_type": error_type,
                "error_code": error_code,
                "error_message": error_msg,
                "full_error": err_json
            }, level="ERROR")
            
            # Handle specific error types
            if http_err.response.status_code == 401:
                st.error("🔑 Invalid API key. Please check your OpenAI API key.")
                return "Authentication failed. Please verify your API key."
            elif http_err.response.status_code == 429:
                st.error("⚠️ OpenAI Rate Limit / Quota Issue")
                st.warning("""
                **OpenAI Free Tier Limits (Most Likely Cause):**
                - Free tier: Only **3 requests per minute (RPM)**
                - You need to wait 60 seconds between requests
                - Your $18 credit is fine - this is about request frequency, not cost
                
                **Solutions:**
                1. Wait 1 minute and try again
                2. Upgrade to Tier 1 ($5 minimum spend) for 500 RPM
                3. Use Google Gemini instead (free tier has better limits)
                
                **Check your tier at:** https://platform.openai.com/account/limits
                """)
                return "Rate limit (3 RPM for free tier). Wait 60 seconds or use Gemini."
            elif http_err.response.status_code == 400:
                st.error(f"❌ Bad request: {error_msg}")
                return f"Request error: {error_msg}"
            else:
                st.error(f"❌ OpenAI API error ({error_code}): {error_msg}")
                return f"API error: {error_msg}"
                
        except Exception as parse_err:
            log_debug(f"Failed to parse error response: {parse_err}", level="ERROR")
            st.error(f"❌ OpenAI HTTP error: {http_err}")
            return f"Request failed with status {http_err.response.status_code}"
    
    except requests.exceptions.ConnectionError as conn_err:
        error_msg = "Failed to connect to OpenAI API. Check your internet connection."
        log_debug(error_msg, {"error": str(conn_err)}, level="ERROR")
        st.error(f"🌐 {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Unexpected OpenAI error: {type(e).__name__}: {str(e)}"
        log_debug(error_msg, {"exception": str(e)}, level="ERROR")
        st.error(f"❌ {error_msg}")
        return "An unexpected error occurred."

# ==================== GEMINI API ====================

def call_gemini_api(prompt: str, api_key: str, context: List[Dict[str, str]]) -> str:
    """Calls Google Gemini API with proper system instruction handling."""
    # Get model from session state or use default
    model_name = st.session_state.get('gemini_model', 'gemini-2.5-flash')
    # FIXED: Use v1 endpoint instead of v1beta and correct URL structure
    api_url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
    
    log_debug(f"Starting Gemini API call with model: {model_name}", level="INFO")
    
    headers = {"Content-Type": "application/json"}
    
    # Extract system instruction
    system_instruction = None
    history = []
    
    # Combine context and the current prompt for processing
    full_context = context + [{"role": "user", "content": prompt}]
    
    # Prepend system instruction to the very first user message in the history
    is_first_user_turn = True
    for msg in full_context:
        if msg["role"] == "system":
            system_instruction = msg["content"]
            log_debug("System instruction found", {"preview": system_instruction[:100]}, level="DEBUG")
            continue
        
        role = "user" if msg["role"] == "user" else "model"
        
        current_content = msg["content"]
        # FIXED: Prepend system instruction to first user message instead of using systemInstruction field
        if role == "user" and is_first_user_turn and system_instruction:
            current_content = f"{system_instruction}\n\n---\n\n{current_content}"
            is_first_user_turn = False
        
        history.append({"role": role, "parts": [{"text": current_content}]})
    
    # Build payload with proper structure
    payload = {"contents": history}
    
    log_debug("Gemini Request Payload", {
        "url": api_url[:100] + "...",  # Don't log full API key
        "contents_count": len(history),
        "has_system_instruction": system_instruction is not None,
        "last_message_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "full_payload": payload
    }, level="DEBUG")
    
    # Store for UI display
    st.session_state['last_ai_payload'] = payload
    st.session_state['last_ai_service'] = 'Gemini'
    
    try:
        with st.spinner("🤖 Gemini is thinking..."):
            start_time = time.time()
            
            log_debug("Sending request to Gemini...", level="INFO")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            elapsed = time.time() - start_time
            log_debug(f"Gemini response received in {elapsed:.2f}s", level="SUCCESS")
            
            # Log response details
            log_debug("Gemini Response Details", {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "elapsed_seconds": elapsed
            }, level="DEBUG")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            log_debug("Gemini Response Data", data, level="DEBUG")
            
            # Check for prompt feedback (blocking)
            if "promptFeedback" in data:
                feedback = data["promptFeedback"]
                if feedback.get("blockReason"):
                    reason = feedback["blockReason"]
                    log_debug(f"Gemini blocked prompt: {reason}", feedback, level="WARNING")
                    st.warning(f"⚠️ Content was blocked: {reason}")
                    return f"Response blocked due to: {reason}"
            
            # Extract response
            if "candidates" not in data or not data["candidates"]:
                log_debug("No candidates in Gemini response", data, level="ERROR")
                return "Gemini returned no response candidates."
            
            candidate = data["candidates"][0]
            
            # Check finish reason
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason not in ["STOP", "MAX_TOKENS"]:
                log_debug(f"Unusual finish reason: {finish_reason}", candidate, level="WARNING")
            
            # Extract text from parts
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                log_debug("No parts in Gemini response", candidate, level="ERROR")
                st.error("❌ Gemini returned empty response")
                st.info("""
                **Possible causes:**
                - Content was filtered/blocked
                - Model doesn't support the request format
                - Try a different model from the sidebar
                
                **Finish reason:** {}
                """.format(finish_reason))
                return "Gemini returned empty response. Check debug logs or try a different model."
            
            text = "".join([p.get("text", "") for p in parts])
            
            if not text or text.strip() == "":
                log_debug("Gemini returned empty text", {"parts": parts, "finish_reason": finish_reason}, level="ERROR")
                st.error(f"❌ Gemini returned empty text (finish reason: {finish_reason})")
                return "Gemini returned empty text. The response may have been filtered."
            
            # Log token usage if available
            if "usageMetadata" in data:
                usage = data["usageMetadata"]
                log_debug(f"Token usage: {usage.get('totalTokenCount', 'N/A')} total "
                         f"({usage.get('promptTokenCount', 'N/A')} prompt + "
                         f"{usage.get('candidatesTokenCount', 'N/A')} response)", 
                         usage, level="INFO")
            
            log_debug(f"Gemini success! Response length: {len(text)} chars", level="SUCCESS")
            
            return text.strip() or "Gemini returned empty text."
            
    except requests.exceptions.Timeout:
        error_msg = "Gemini request timed out after 60 seconds"
        log_debug(error_msg, level="ERROR")
        st.error(f"⏱️ {error_msg}")
        return "Request timed out. The API might be slow or unavailable."
        
    except requests.exceptions.HTTPError as http_err:
        log_debug(f"Gemini HTTP error: {http_err}", level="ERROR")
        
        try:
            err_json = http_err.response.json()
            error_details = err_json.get("error", {})
            error_msg = error_details.get("message", str(http_err))
            error_code = error_details.get("code", http_err.response.status_code)
            
            log_debug("Gemini Error Details", {
                "status_code": http_err.response.status_code,
                "error_code": error_code,
                "error_message": error_msg,
                "full_error": err_json
            }, level="ERROR")
            
            # Handle specific errors
            if http_err.response.status_code == 400:
                if "API_KEY_INVALID" in str(err_json):
                    st.error("🔑 Invalid API key. Please check your Gemini API key.")
                    return "Invalid API key. Please verify your Gemini API key."
                else:
                    st.error(f"❌ Bad request: {error_msg}")
                    return f"Request error: {error_msg}"
            elif http_err.response.status_code == 404:
                st.error("❌ Model Not Found")
                st.warning(f"""
                **The model endpoint was not found.**
                
                Current model: `{st.session_state.get('gemini_model', 'unknown')}`
                
                **Available models:**
                - gemini-2.5-flash (recommended - latest)
                - gemini-2.0-flash-exp
                - gemini-1.5-pro
                - gemini-1.5-flash
                
                Try selecting a different model from the sidebar.
                """)
                return "Model not found. Please select a different Gemini model."
            elif http_err.response.status_code == 429:
                st.error("⚠️ Rate limit exceeded or quota depleted.")
                return "Rate limit exceeded. Please wait or check your Gemini quota."
            else:
                st.error(f"❌ Gemini API error ({error_code}): {error_msg}")
                return f"API error: {error_msg}"
                
        except Exception as parse_err:
            log_debug(f"Failed to parse error response: {parse_err}", level="ERROR")
            st.error(f"❌ Gemini HTTP error: {http_err}")
            return f"Request failed with status {http_err.response.status_code}"
    
    except requests.exceptions.ConnectionError as conn_err:
        error_msg = "Failed to connect to Gemini API. Check your internet connection."
        log_debug(error_msg, {"error": str(conn_err)}, level="ERROR")
        st.error(f"🌐 {error_msg}")
        return error_msg
        
    except Exception as e:
        error_msg = f"Unexpected Gemini error: {type(e).__name__}: {str(e)}"
        log_debug(error_msg, {"exception": str(e)}, level="ERROR")
        st.error(f"❌ {error_msg}")
        return "An unexpected error occurred."

# ==================== PROMPT GENERATION ====================

def list_gemini_models(api_key: str):
    """Calls the Gemini API to list available models and displays them."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        with st.spinner("Fetching available Gemini models..."):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            models = response.json().get("models", [])
            
            supported_models = []
            for model in models:
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    supported_models.append({
                        "Model Name": model.get("name"),
                        "Display Name": model.get("displayName"),
                        "Description": model.get("description")
                    })
            
            if supported_models:
                st.session_state['gemini_models'] = pd.DataFrame(supported_models)
                log_debug(f"Found {len(supported_models)} Gemini models", level="SUCCESS")
            else:
                st.session_state['gemini_models_error'] = "No models supporting 'generateContent' found."
                log_debug("No compatible Gemini models found", level="WARNING")

    except requests.exceptions.HTTPError as http_err:
        try:
            msg = http_err.response.json().get("error", {}).get("message", str(http_err))
            st.session_state['gemini_models_error'] = f"Failed to list models: {msg}"
            log_debug(f"Failed to list Gemini models: {msg}", level="ERROR")
        except Exception:
            st.session_state['gemini_models_error'] = f"Failed to list models: {http_err}"
            log_debug(f"Failed to list Gemini models: {http_err}", level="ERROR")
    except Exception as e:
        st.session_state['gemini_models_error'] = f"An unexpected error occurred: {e}"
        log_debug(f"Unexpected error listing models: {e}", level="ERROR")

def generate_summary_prompt(clingen_data: Dict, myvariant_data: Dict, vep_data: List) -> str:
    """Creates a detailed prompt for AI to summarize variant data."""
    
    # Simplify myvariant data to reduce token usage
    if myvariant_data and 'dbnsfp' in myvariant_data:
        myvariant_data['dbnsfp'] = {
            k: v for k, v in myvariant_data['dbnsfp'].items()
            if k in ['sift', 'polyphen2_hdiv', 'polyphen2_hvar', 'cadd', 'revel', 'gerp++_rs']
        }
    
    summary_instruction = """
    Please provide a comprehensive but clear summary of the genetic variant data provided below. 
    
    Organize your summary into these sections:
    
    1. **Variant Identification**: State key identifiers (HGVS, RSID, ClinGen Allele ID).
    
    2. **Clinical Significance**: Detail ClinVar findings (significance, review status, conditions).
    
    3. **Population Frequencies**: Report the highest overall gnomAD allele frequency and its source. 
       Note if the variant is common, rare, etc.
    
    4. **Functional Predictions**: Summarize SIFT and PolyPhen predictions (score and qualitative prediction).
    
    5. **Transcript/Gene Consequences**: Describe the most significant VEP molecular consequence, 
       affected gene, and impact level.
    
    **Rules**: 
    - Be accurate and traceable to the source data
    - Be clear and concise
    - Use bullet points where appropriate
    """
    
    data_parts = [summary_instruction]
    
    if clingen_data:
        data_parts.append(f"\n**ClinGen Data:**\n```json\n{json.dumps(clingen_data, indent=2)}\n```")
    
    if myvariant_data:
        data_parts.append(f"\n**MyVariant.info Data:**\n```json\n{json.dumps(myvariant_data, indent=2)}\n```")
    
    if vep_data:
        data_parts.append(f"\n**Ensembl VEP Data:**\n```json\n{json.dumps(vep_data, indent=2)}\n```")
    
    return "\n\n".join(data_parts)

# ==================== AI ASSISTANT UI ====================

def display_ai_assistant(analysis_data: Optional[Dict]):
    """Renders the AI Assistant UI with debug panel."""
    st.markdown('<div class="section-header">🤖 AI Assistant</div>', unsafe_allow_html=True)
    
    # Show debug panel
    show_debug_panel()
    
    # AI service selection
    st.sidebar.markdown("### 🤖 AI Assistant Settings")
    ai_service = st.sidebar.radio("Select AI Service", ('OpenAI', 'Google Gemini'))
    
    # Model selection for Gemini
    if ai_service == 'Google Gemini':
        gemini_model = st.sidebar.selectbox(
            "Gemini Model",
            ['gemini-2.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro'],
            index=0,
            help="Gemini 2.5 Flash is recommended for best performance"
        )
        st.session_state['gemini_model'] = gemini_model
        
        # Add model list button

    
    # Get API key
    api_key = get_manual_api_key(ai_service)
    if not api_key:
        st.info("💡 The AI Assistant is unavailable until an API key is provided.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        log_debug("Initialized new chat session", level="INFO")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show last request payload
    if 'last_ai_payload' in st.session_state:
        with st.expander(f"🔍 Last {st.session_state.get('last_ai_service', 'AI')} Request Payload"):
            st.json(st.session_state['last_ai_payload'])
    
    # Show Gemini models if available
    if ai_service == 'Google Gemini':
        if 'gemini_models' in st.session_state:
            with st.expander("✅ Available Gemini Models"):
                st.dataframe(st.session_state['gemini_models'], use_container_width=True)
        elif 'gemini_models_error' in st.session_state:
            with st.expander("❌ Model List Error"):
                st.error(st.session_state['gemini_models_error'])
    
    # System prompt
    system_prompt = {
        "role": "system",
        "content": """You are a knowledgeable assistant specializing in genomics and bioinformatics. 
Help users understand genetic variant data. When summarizing, be accurate, cite data sources, 
and be traceable. You can also answer general questions about genetics and genomics."""
    }
    
    # Summarize button
    if analysis_data:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📝 Summarize & Interpret Results", key="summarize_ai", use_container_width=True):
                log_debug("User requested summary", level="INFO")
                
                prompt = generate_summary_prompt(
                    analysis_data.get('clingen_data'),
                    analysis_data.get('annotations', {}).get('myvariant_data'),
                    analysis_data.get('annotations', {}).get('vep_data')
                )
                
                st.session_state.messages.append({
                    "role": "user",
                    "content": "Please summarize and interpret the results."
                })
                
                # Call appropriate API
                if ai_service == 'OpenAI':
                    response = call_openai_api(prompt, api_key, context=[system_prompt])
                else:
                    response = call_gemini_api(prompt, api_key, context=[system_prompt])
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the results or genetics in general..."):
        log_debug(f"User query: {prompt[:100]}...", level="INFO")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Call appropriate API
        if ai_service == 'OpenAI':
            ai_response = call_openai_api(
                prompt,
                api_key,
                context=[system_prompt] + st.session_state.messages
            )
        else:
            ai_response = call_gemini_api(
                prompt,
                api_key,
                context=[system_prompt] + st.session_state.messages
            )
        
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response
        })

# ==================== CLINGEN FUNCTIONS ====================

def query_clingen_allele(hgvs: str) -> Dict[str, Any]:
    """Query ClinGen Allele Registry by HGVS notation."""
    base_url = "https://reg.clinicalgenome.org/allele"
    params = {'hgvs': hgvs}
    
    with st.spinner(f"Querying ClinGen for: {hgvs}"):
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

def parse_caid_minimal(raw_json):
    """Parse ClinGen Allele Registry JSON to extract key information."""
    result = {}
    result['CAid'] = raw_json.get('@id', '').split('/')[-1]
    dbsnp = raw_json.get('externalRecords', {}).get('dbSNP', [])
    result['rsid'] = dbsnp[0].get('rs') if dbsnp else None
    genomic = raw_json.get('genomicAlleles', [])
    result['genomic_hgvs_grch38'] = None
    result['genomic_hgvs_grch37'] = None
    for g in genomic:
        hgvs_list = g.get('hgvs', [])
        ref_genome = g.get('referenceGenome', '')
        if 'GRCh38' in ref_genome and hgvs_list:
            result['genomic_hgvs_grch38'] = hgvs_list[0]
        elif 'GRCh37' in ref_genome and hgvs_list:
            result['genomic_hgvs_grch37'] = hgvs_list[0]
    myv = raw_json.get('externalRecords', {})
    result['myvariant_hg38'] = myv.get('MyVariantInfo_hg38', [{}])[0].get('id') if myv.get('MyVariantInfo_hg38') else None
    result['myvariant_hg19'] = myv.get('MyVariantInfo_hg19', [{}])[0].get('id') if myv.get('MyVariantInfo_hg19') else None
    result['mane_ensembl'] = None
    result['mane_refseq'] = None
    transcripts = raw_json.get('transcriptAlleles', [])
    for t in transcripts:
        mane = t.get('MANE', {})
        if mane and mane.get('maneStatus') == 'MANE Select':
            if 'nucleotide' in mane:
                ensembl_info = mane['nucleotide'].get('Ensembl', {})
                refseq_info = mane['nucleotide'].get('RefSeq', {})
                result['mane_ensembl'] = ensembl_info.get('hgvs')
                result['mane_refseq'] = refseq_info.get('hgvs')
            break
    return result

# ==================== VARIANT ANNOTATION FUNCTIONS ====================

def get_variant_annotations(clingen_data, classification=None):
    """Retrieve variant annotations from multiple APIs."""
    annotations = {'myvariant_data': {}, 'vep_data': [], 'errors': []}
    query_id = None
    if clingen_data.get('myvariant_hg38'):
        query_id = clingen_data['myvariant_hg38']
    elif classification and classification.query_type == 'rsid':
        query_id = classification.extracted_identifier
    if query_id:
        try:
            with st.spinner("Fetching MyVariant.info data..."):
                myv_url = f"https://myvariant.info/v1/variant/{query_id}?assembly=hg38"
                myv_response = requests.get(myv_url, timeout=30)
                if myv_response.ok:
                    myv_raw = myv_response.json()
                    if isinstance(myv_raw, list) and len(myv_raw) > 0:
                        myv_raw = myv_raw[0]
                    annotations['myvariant_data'] = myv_raw
                else:
                    annotations['errors'].append(f"MyVariant query failed: HTTP {myv_response.status_code}")
        except Exception as e:
            annotations['errors'].append(f"MyVariant query error: {str(e)}")
    vep_input = None
    vep_attempted = False
    if clingen_data.get('mane_ensembl'):
        vep_input = clingen_data['mane_ensembl']
        vep_attempted = True
        try:
            with st.spinner("Fetching Ensembl VEP data..."):
                vep_url = f"https://rest.ensembl.org/vep/human/hgvs/{vep_input}"
                vep_headers = {"Content-Type": "application/json", "Accept": "application/json"}
                vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                if vep_response.ok:
                    annotations['vep_data'] = vep_response.json()
                else:
                    annotations['errors'].append(f"VEP query with MANE transcript failed: HTTP {vep_response.status_code}")
        except Exception as e:
            annotations['errors'].append(f"VEP query with MANE transcript error: {str(e)}")
    if (classification and classification.query_type == 'rsid' and not annotations['vep_data'] and not vep_attempted):
        vep_input = classification.extracted_identifier
        vep_attempted = True
        try:
            with st.spinner("Fetching Ensembl VEP data with RSID..."):
                vep_url = f"https://rest.ensembl.org/vep/human/hgvs/{vep_input}"
                vep_headers = {"Content-Type": "application/json", "Accept": "application/json"}
                vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                if vep_response.ok:
                    annotations['vep_data'] = vep_response.json()
                else:
                    annotations['errors'].append(f"VEP query with RSID failed: HTTP {vep_response.status_code}")
        except Exception as e:
            annotations['errors'].append(f"VEP query with RSID error: {str(e)}")
    if (not annotations['vep_data'] and annotations['myvariant_data'] and isinstance(annotations['myvariant_data'], dict)):
        dbnsfp = annotations['myvariant_data'].get('dbnsfp', {})
        ensembl_data = dbnsfp.get('ensembl', {})
        transcript_ids = ensembl_data.get('transcriptid', [])
        if transcript_ids:
            if isinstance(transcript_ids, list) and len(transcript_ids) > 0:
                primary_transcript = transcript_ids[0]
            else:
                primary_transcript = transcript_ids
            hgvs_coding = dbnsfp.get('hgvsc')
            if hgvs_coding:
                if isinstance(hgvs_coding, list):
                    hgvs_coding = hgvs_coding[0]
                vep_hgvs = f"{primary_transcript}:{hgvs_coding}"
                try:
                    with st.spinner(f"Fetching VEP data with Ensembl transcript {primary_transcript}..."):
                        vep_url = f"https://rest.ensembl.org/vep/human/hgvs/{vep_hgvs}"
                        vep_headers = {"Content-Type": "application/json", "Accept": "application/json"}
                        vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                        if vep_response.ok:
                            annotations['vep_data'] = vep_response.json()
                            annotations['vep_fallback_used'] = True
                            st.success(f"VEP fallback successful using transcript {primary_transcript}")
                        else:
                            annotations['errors'].append(f"VEP fallback query failed: HTTP {vep_response.status_code}")
                except Exception as e:
                    annotations['errors'].append(f"VEP fallback query error: {str(e)}")
    return annotations

# ==================== VEP DISPLAY FUNCTIONS ====================

def select_primary_vep_transcript(vep_data):
    """Select the primary transcript for VEP analysis based on priority."""
    if not vep_data or not vep_data[0].get('transcript_consequences'):
        return None, "No transcript consequences found"
    transcripts = vep_data[0]['transcript_consequences']
    for t in transcripts:
        flags = t.get('flags', [])
        if 'MANE_SELECT' in flags or any('mane' in str(flag).lower() for flag in flags):
            return t, "MANE Select"
    for t in transcripts:
        if t.get('canonical') == 1 or 'canonical' in t.get('flags', []):
            return t, "Canonical"
    for t in transcripts:
        if (t.get('biotype') == 'protein_coding' and 'missense_variant' in t.get('consequence_terms', [])):
            return t, "First protein coding with missense annotation"
    for t in transcripts:
        if t.get('biotype') == 'protein_coding':
            return t, "First protein coding"
    return transcripts[0], "First available transcript"

def display_vep_analysis(vep_data):
    """Display comprehensive VEP analysis."""
    if not vep_data or not vep_data[0].get('transcript_consequences'):
        st.warning("No VEP data available")
        return
    variant_info = vep_data[0]
    all_transcripts = variant_info.get('transcript_consequences', [])
    primary_transcript, selection_reason = select_primary_vep_transcript(vep_data)
    if primary_transcript:
        st.subheader(f"Primary Transcript Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Transcript:** {primary_transcript.get('transcript_id', 'N/A')}")
            st.write(f"**Gene:** {primary_transcript.get('gene_symbol', 'N/A')} ({primary_transcript.get('gene_id', 'N/A')})")
        with col2:
            st.info(f"**Selection Criteria:** {selection_reason}")
        col1, col2, col3 = st.columns(3)
        with col1:
            consequences = primary_transcript.get('consequence_terms', [])
            st.write(f"**Consequence:** {', '.join(consequences)}")
        with col2:
            st.write(f"**Impact:** {primary_transcript.get('impact', 'N/A')}")
        with col3:
            st.write(f"**Biotype:** {primary_transcript.get('biotype', 'N/A')}")
        if primary_transcript.get('amino_acids'):
            st.subheader("Sequence Changes")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Amino Acid Change:** {primary_transcript.get('amino_acids', 'N/A')}")
                st.write(f"**Position:** {primary_transcript.get('protein_start', 'N/A')}")
            with col2:
                st.write(f"**Codon Change:** {primary_transcript.get('codons', 'N/A')}")
                st.write(f"**CDS Position:** {primary_transcript.get('cds_start', 'N/A')}")
            with col3:
                st.write(f"**cDNA Position:** {primary_transcript.get('cdna_start', 'N/A')}")
        if primary_transcript.get('sift_score') or primary_transcript.get('polyphen_score'):
            st.subheader("Functional Predictions")
            col1, col2 = st.columns(2)
            with col1:
                if primary_transcript.get('sift_score'):
                    st.metric("SIFT Score", f"{primary_transcript['sift_score']:.3f}")
                    st.write(f"**SIFT Prediction:** {primary_transcript.get('sift_prediction', 'N/A')}")
            with col2:
                if primary_transcript.get('polyphen_score'):
                    st.metric("PolyPhen Score", f"{primary_transcript['polyphen_score']:.3f}")
                    st.write(f"**PolyPhen Prediction:** {primary_transcript.get('polyphen_prediction', 'N/A')}")
    with st.expander(f"View All {len(all_transcripts)} Transcripts", expanded=False):
        for i, transcript in enumerate(all_transcripts, 1):
            with st.container():
                st.markdown(f"### Transcript {i}: {transcript.get('transcript_id', 'N/A')}")
                flags = transcript.get('flags', [])
                special_flags = []
                if transcript.get('canonical') == 1: special_flags.append("CANONICAL")
                if 'MANE_SELECT' in flags: special_flags.append("MANE SELECT")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Gene:** {transcript.get('gene_symbol', 'N/A')}")
                    if special_flags: st.success(f"🏷️ {', '.join(special_flags)}")
                with col2:
                    st.write(f"**Consequence:** {', '.join(transcript.get('consequence_terms', []))}")
                    st.write(f"**Impact:** {transcript.get('impact', 'N/A')}")
                with col3:
                    st.write(f"**Biotype:** {transcript.get('biotype', 'N/A')}")
                    if transcript.get('distance'): st.write(f"**Distance:** {transcript.get('distance', 'N/A')}")
                with col4:
                    if transcript.get('amino_acids'):
                        st.write(f"**AA Change:** {transcript.get('amino_acids', 'N/A')}")
                        st.write(f"**Position:** {transcript.get('protein_start', 'N/A')}")
                if transcript.get('sift_score') or transcript.get('polyphen_score'):
                    pred_col1, pred_col2 = st.columns(2)
                    with pred_col1:
                        if transcript.get('sift_score'): st.write(f"**SIFT:** {transcript['sift_score']:.3f} ({transcript.get('sift_prediction', 'N/A')})")
                    with pred_col2:
                        if transcript.get('polyphen_score'): st.write(f"**PolyPhen:** {transcript['polyphen_score']:.3f} ({transcript.get('polyphen_prediction', 'N/A')})")
                st.markdown("---")

# ==================== MYVARIANT DISPLAY FUNCTIONS ====================

def display_comprehensive_myvariant_data(myvariant_data):
    """Display comprehensive MyVariant.info data analysis."""
    if not myvariant_data:
        st.warning("No MyVariant data available")
        return
    
    if isinstance(myvariant_data, list):
        if len(myvariant_data) > 1: st.info(f"Multiple variants found ({len(myvariant_data)}). Showing first result.")
        if len(myvariant_data) > 0: myvariant_data = myvariant_data[0]
        else:
            st.warning("Empty response from MyVariant")
            return
    if not isinstance(myvariant_data, dict):
        st.error("Unexpected data format from MyVariant")
        return
    
    data_tabs = st.tabs(["🧬 Basic Info", "🔬 Functional Predictions", "📊 Population Frequencies", "🏥 ClinVar", "🔗 External DBs"])
    
    with data_tabs[0]:  # Basic Info
        st.subheader("Variant Information")
        col1, col2, col3 = st.columns(3)
        chrom = (myvariant_data.get('hg38', {}).get('chr') or myvariant_data.get('chrom') or 'N/A')
        hg38_data = myvariant_data.get('hg38', {})
        pos = (hg38_data.get('start') or hg38_data.get('end') or hg38_data.get('pos') or myvariant_data.get('pos') or myvariant_data.get('vcf', {}).get('position') or 'N/A')
        ref = (myvariant_data.get('hg38', {}).get('ref') or myvariant_data.get('ref') or myvariant_data.get('vcf', {}).get('ref') or 'N/A')
        alt = (myvariant_data.get('hg38', {}).get('alt') or myvariant_data.get('alt') or myvariant_data.get('vcv', {}).get('alt') or 'N/A')
        with col1:
            st.write(f"**Chromosome:** {chrom}")
            st.write(f"**Position (hg38):** {pos}")
        with col2:
            st.write(f"**Reference:** {ref}")
            st.write(f"**Alternate:** {alt}")
        with col3:
            gene_name = (myvariant_data.get('clinvar', {}).get('gene', {}).get('symbol') or 
                         myvariant_data.get('snpeff', {}).get('ann', [{}])[0].get('genename') or 
                         (myvariant_data.get('dbnsfp', {}).get('genename') if isinstance(myvariant_data.get('dbnsfp', {}).get('genename'), str) else None) or
                         (myvariant_data.get('dbnsfp', {}).get('genename', [None])[0] if isinstance(myvariant_data.get('dbnsfp', {}).get('genename'), list) else None) or 'N/A')

            st.write(f"**Gene:** {gene_name}")
            rsid = myvariant_data.get('rsid') or myvariant_data.get('dbsnp', {}).get('rsid') or 'N/A'
            st.write(f"**RSID:** {rsid}")
        if myvariant_data.get('clingen'):
            st.subheader("ClinGen Information")
            clingen = myvariant_data['clingen']
            st.write(f"**CAID:** {clingen.get('caid', 'N/A')}")
    
    with data_tabs[1]:  # Functional Predictions
        st.subheader("Functional Prediction Scores")
        dbnsfp = myvariant_data.get('dbnsfp', {})
        if not dbnsfp:
            st.info("No dbNSFP functional prediction data available")
            return
        
        def extract_nested_value(data, path_list):
            current = data
            for key in path_list:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        
        prediction_categories = {
            "Pathogenicity Predictors": [("SIFT", ["sift", "score"], ["sift", "pred"]), ("PolyPhen2 HDiv", ["polyphen2", "hdiv", "score"], ["polyphen2", "hdiv", "pred"]), ("PolyPhen2 HVar", ["polyphen2", "hvar", "score"], ["polyphen2", "hvar", "pred"]), ("FATHMM", ["fathmm", "score"], ["fathmm", "pred"]), ("MutationTaster", ["mutationtaster", "score"], ["mutationtaster", "pred"]), ("MutationAssessor", ["mutationassessor", "score"], ["mutationassessor", "pred"]), ("PROVEAN", ["provean", "score"], ["provean", "pred"]), ("MetaSVM", ["metasvm", "score"], ["metasvm", "pred"]), ("MetaLR", ["metalr", "score"], ["metalr", "pred"]), ("M-CAP", ["m-cap", "score"], ["m-cap", "pred"]), ("REVEL", ["revel", "score"], None), ("MutPred", ["mutpred", "score"], None), ("LRT", ["lrt", "score"], ["lrt", "pred"])],
            "Conservation Scores": [("GERP++ NR", ["gerp++", "nr"], None), ("GERP++ RS", ["gerp++", "rs"], None), ("PhyloP 100way Vertebrate", ["phylop", "100way_vertebrate", "score"], None), ("PhyloP 470way Mammalian", ["phylop", "470way_mammalian", "score"], None), ("PhastCons 100way Vertebrate", ["phastcons", "100way_vertebrate", "score"], None), ("PhastCons 470way Mammalian", ["phastcons", "470way_mammalian", "score"], None), ("SiPhy 29way", ["siphy_29way", "logodds_score"], None)],
            "Ensemble Predictors": [("CADD Phred", ["cadd", "phred"], None), ("DANN", ["dann", "score"], None), ("Eigen PC Phred", ["eigen-pc", "phred_coding"], None), ("FATHMM-MKL", ["fathmm-mkl", "coding_score"], ["fathmm-mkl", "coding_pred"]), ("FATHMM-XF", ["fathmm-xf", "coding_score"], ["fathmm-xf", "coding_pred"]), ("GenoCanyon", ["genocanyon", "score"], None), ("Integrated FitCons", ["fitcons", "integrated", "score"], None), ("VEST4", ["vest4", "score"], None), ("MVP", ["mvp", "score"], None)],
            "Deep Learning": [("PrimateAI", ["primateai", "score"], ["primateai", "pred"]), ("DEOGEN2", ["deogen2", "score"], ["deogen2", "pred"]), ("BayesDel AddAF", ["bayesdel", "add_af", "score"], ["bayesdel", "add_af", "pred"]), ("ClinPred", ["clinpred", "score"], ["clinpred", "pred"]), ("LIST-S2", ["list-s2", "score"], ["list-s2", "pred"]), ("AlphaMissense", ["alphamissense", "score"], ["alphamissense", "pred"]), ("ESM1b", ["esm1b", "score"], ["esm1b", "pred"])]
        }
        
        for category, predictors in prediction_categories.items():
            st.markdown(f"#### {category}")
            predictor_data = []
            for predictor_info in predictors:
                if len(predictor_info) == 3: predictor_name, score_path, pred_path = predictor_info
                else: continue
                score_val = extract_nested_value(dbnsfp, score_path)
                if isinstance(score_val, list) and score_val: score_val = score_val[0]
                pred_val = None
                if pred_path:
                    pred_val = extract_nested_value(dbnsfp, pred_path)
                    if isinstance(pred_val, list) and pred_val: pred_val = pred_val[0]
                if score_val is not None:
                    predictor_data.append({'Predictor': predictor_name, 'Score': score_val, 'Prediction': pred_val or 'N/A'})
            if predictor_data:
                cols = st.columns(3)
                for i, pred in enumerate(predictor_data):
                    with cols[i % 3]:
                        score_str = f"{pred['Score']:.3f}" if isinstance(pred['Score'], float) else str(pred['Score'])
                        prediction_text = pred['Prediction']
                        if prediction_text in ['D', 'Damaging', 'DAMAGING']: prediction_color = "🔴"
                        elif prediction_text in ['T', 'Tolerated', 'TOLERATED', 'B', 'Benign']: prediction_color = "🟢" 
                        elif prediction_text in ['P', 'Possibly damaging', 'POSSIBLY_DAMAGING']: prediction_color = "🟡"
                        else: prediction_color = ""
                        display_pred = f"{prediction_color} {prediction_text}" if prediction_color else prediction_text
                        st.metric(pred['Predictor'], score_str, delta=display_pred if display_pred != 'N/A' else None)
            else:
                st.info(f"No {category.lower()} data available")

    with data_tabs[2]: # Population Frequencies
        st.subheader("Population Frequency Data")
        freq_tabs = st.tabs(["gnomAD Exome", "gnomAD Genome", "1000 Genomes", "ExAC", "Raw Data"])
        
        with freq_tabs[0]:
            gnomad_exome = myvariant_data.get('gnomad_exome', {})
            if gnomad_exome:
                st.markdown("**gnomAD Exome v2.1.1**")
                af_data, an_data, ac_data = gnomad_exome.get('af', {}), gnomad_exome.get('an', {}), gnomad_exome.get('ac', {})
                if isinstance(af_data, dict):
                    pop_data = []
                    populations = {'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 'af_asj': 'Ashkenazi Jewish', 'af_eas': 'East Asian', 'af_fin': 'Finnish', 'af_nfe': 'Non-Finnish European', 'af_sas': 'South Asian', 'af_oth': 'Other'}
                    for pop_key, pop_name in populations.items():
                        freq = af_data.get(pop_key)
                        if freq is not None and freq > 0 and freq <= st.session_state.get('freq_threshold', 1.0):
                            an, ac = an_data.get(pop_key.replace('af', 'an')), ac_data.get(pop_key.replace('af', 'ac'))
                            pop_data.append({'Population': pop_name, 'Frequency': freq, 'Allele Count': ac or 'N/A', 'Total Alleles': an or 'N/A'})
                    if pop_data:
                        df_freq = pd.DataFrame(pop_data).sort_values(by="Frequency", ascending=False)
                        st.dataframe(df_freq, use_container_width=True)
                        chart_data = df_freq.set_index('Population')['Frequency']
                        if not chart_data.empty: st.bar_chart(chart_data)
                    else:
                        st.info("No gnomAD exome populations match the current filter settings.")
                else: st.info("gnomAD exome data format not recognized.")
            else: st.info("No gnomAD exome data available.")
        
        with freq_tabs[1]:
            gnomad_genome = myvariant_data.get('gnomad_genome', {})
            if gnomad_genome:
                st.markdown("**gnomAD Genome v3.1.2**")
                af_data, an_data, ac_data = gnomad_genome.get('af', {}), gnomad_genome.get('an', {}), gnomad_genome.get('ac', {})
                if isinstance(af_data, dict):
                    pop_data = []
                    populations = {'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 'af_ami': 'Amish', 'af_asj': 'Ashkenazi Jewish', 'af_eas': 'East Asian', 'af_fin': 'Finnish', 'af_mid': 'Middle Eastern', 'af_nfe': 'Non-Finnish European', 'af_sas': 'South Asian', 'af_oth': 'Other'}
                    for pop_key, pop_name in populations.items():
                        freq = af_data.get(pop_key)
                        if freq is not None and freq > 0 and freq <= st.session_state.get('freq_threshold', 1.0):
                            an, ac = an_data.get(pop_key.replace('af', 'an')), ac_data.get(pop_key.replace('af', 'ac'))
                            pop_data.append({'Population': pop_name, 'Frequency': freq, 'Allele Count': ac or 'N/A', 'Total Alleles': an or 'N/A'})
                    if pop_data:
                        df_freq = pd.DataFrame(pop_data).sort_values(by="Frequency", ascending=False)
                        st.dataframe(df_freq, use_container_width=True)
                        chart_data = df_freq.set_index('Population')['Frequency']
                        if not chart_data.empty: st.bar_chart(chart_data)
                    else: st.info("No gnomAD genome populations match the current filter settings.")
        
        with freq_tabs[2]: # 1000 Genomes
            kg_data = myvariant_data.get('dbnsfp', {}).get('1000gp3', {})
            if kg_data:
                st.markdown("**1000 Genomes Project Phase 3**")
                if isinstance(kg_data, list): kg_data = kg_data[0]
                
                pop_data = []
                populations = {'af':'Global', 'afr_af':'African', 'amr_af':'American', 'eas_af':'East Asian', 'eur_af':'European', 'sas_af':'South Asian'}
                for key, name in populations.items():
                    actual_key = key.split('_')[0] if '_' in key else 'af'
                    freq_data = kg_data.get(actual_key)
                    
                    freq = None
                    if isinstance(freq_data, dict):
                        freq = freq_data.get('af')
                    elif actual_key == 'af':
                        freq = kg_data.get('af')

                    if freq is not None and freq > 0 and freq <= st.session_state.get('freq_threshold', 1.0):
                        pop_data.append({'Population': name, 'Frequency': freq})
                
                if pop_data:
                    st.dataframe(pd.DataFrame(pop_data), use_container_width=True)
                else:
                    st.info("No 1000 Genomes populations match the current filter.")
            else:
                st.info("No 1000 Genomes data available.")

        with freq_tabs[3]: # ExAC
            exac_data = myvariant_data.get('exac', {}) or myvariant_data.get('dbnsfp', {}).get('exac', {})

            if exac_data:
                st.markdown("**Exome Aggregation Consortium (ExAC)**")
                if isinstance(exac_data, list): exac_data = exac_data[0]
                pop_data = []
                populations = {'af':'Global', 'afr':'African', 'amr':'Latino', 'eas':'East Asian', 'fin':'Finnish', 'nfe':'Non-Finnish European', 'sas':'South Asian', 'oth':'Other'}
                for key, name in populations.items():
                    freq_val = exac_data.get(key)
                    freq = None
                    if isinstance(freq_val, dict):
                        freq = freq_val.get('af')
                    elif isinstance(freq_val, float):
                        freq = freq_val
                    
                    if freq is not None and freq > 0 and freq <= st.session_state.get('freq_threshold', 1.0):
                        pop_data.append({'Population': name, 'Frequency': freq})
                if pop_data:
                    st.dataframe(pd.DataFrame(pop_data), use_container_width=True)
                else:
                    st.info("No ExAC populations match the current filter.")
            else:
                st.info("No ExAC data available.")
        
        with freq_tabs[4]: # Raw Data
            st.markdown("**All Available Frequency Fields**")
            freq_fields = {}
            def collect_freq_fields(data, prefix=""):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if 'af' in key.lower() or 'freq' in key.lower():
                        if isinstance(value, (int, float)) and value > 0 and value <= st.session_state.get('freq_threshold', 1.0):
                            freq_fields[full_key] = value
                    elif isinstance(value, dict):
                        collect_freq_fields(value, full_key)
            collect_freq_fields(myvariant_data)
            if freq_fields:
                st.dataframe(pd.DataFrame([{'Field': k, 'Frequency': v} for k,v in sorted(freq_fields.items())]), use_container_width=True)
            else:
                st.info("No frequency fields match the current filter.")

    with data_tabs[3]: # ClinVar
        st.subheader("ClinVar Clinical Annotations")
        clinvar_data = myvariant_data.get('clinvar', {})
        if not clinvar_data:
            st.info("No ClinVar data available")
            return
        clinical_sig = (clinvar_data.get('clinical_significance') or clinvar_data.get('clnsig') or 'N/A')
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Clinical Significance:** {clinical_sig}")
            if clinvar_data.get('variant_id'): st.write(f"**Variation ID:** {clinvar_data['variant_id']}")
            if clinvar_data.get('allele_id'): st.write(f"**Allele ID:** {clinvar_data['allele_id']}")
        with col2:
            gene_info = clinvar_data.get('gene', {})
            if isinstance(gene_info, dict):
                if gene_info.get('symbol'): st.write(f"**Gene Symbol:** {gene_info['symbol']}")
                if gene_info.get('id'): st.write(f"**Gene ID:** {gene_info['id']}")
        hgvs_info = clinvar_data.get('hgvs', {})
        if isinstance(hgvs_info, dict):
            st.subheader("HGVS Notations")
            col1, col2 = st.columns(2)
            with col1:
                if hgvs_info.get('coding'): st.write(f"**Coding:** {hgvs_info['coding']}")
                if hgvs_info.get('protein'): st.write(f"**Protein:** {hgvs_info['protein']}")
            with col2:
                if hgvs_info.get('genomic'):
                    genomic = hgvs_info['genomic']
                    st.write(f"**Genomic:** {', '.join(genomic) if isinstance(genomic, list) else str(genomic)}")
        rcv_data = clinvar_data.get('rcv', [])
        if rcv_data and isinstance(rcv_data, list):
            st.subheader(f"ClinVar Records ({len(rcv_data)} records)")
            for i, rcv in enumerate(rcv_data, 1):
                if isinstance(rcv, dict):
                    with st.expander(f"Record {i}: {rcv.get('accession', 'N/A')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Accession:** {rcv.get('accession', 'N/A')}")
                            st.write(f"**Clinical Significance:** {rcv.get('clinical_significance', 'N/A')}")
                            st.write(f"**Review Status:** {rcv.get('review_status', 'N/A')}")
                            st.write(f"**Origin:** {rcv.get('origin', 'N/A')}")
                        with col2:
                            st.write(f"**Last Evaluated:** {rcv.get('last_evaluated', 'N/A')}")
                            st.write(f"**Number of Submitters:** {rcv.get('number_submitters', 'N/A')}")
                            conditions = rcv.get('conditions', {})
                            if isinstance(conditions, dict) and conditions.get('name'):
                                st.write(f"**Condition:** {conditions['name']}")
                                identifiers = conditions.get('identifiers', {})
                                if identifiers:
                                    id_list = [f"{db}: {id_val}" for db, id_val in identifiers.items()]
                                    st.write(f"**Identifiers:** {', '.join(id_list)}")
    
    with data_tabs[4]: # External DBs
        st.subheader("External Database References")
        dbsnp_data = myvariant_data.get('dbsnp', {})
        if dbsnp_data:
            st.markdown("#### dbSNP")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**RSID:** {dbsnp_data.get('rsid', 'N/A')}")
                st.write(f"**Build:** {dbsnp_data.get('dbsnp_build', 'N/A')}")
                st.write(f"**Variant Type:** {dbsnp_data.get('vartype', 'N/A')}")
            genes = dbsnp_data.get('gene', [])
            if genes:
                with col2:
                    st.write(f"**Associated Genes:** {len(genes)} genes")
                    for gene in genes[:3]:
                        st.write(f"- {gene.get('symbol', 'N/A')} (ID: {gene.get('geneid', 'N/A')})")
        uniprot_data = myvariant_data.get('uniprot', {})
        if uniprot_data:
            st.markdown("#### UniProt")
            if uniprot_data.get('clinical_significance'): st.write(f"**Clinical Significance:** {uniprot_data['clinical_significance']}")
            if uniprot_data.get('source_db_id'): st.write(f"**Source DB ID:** {uniprot_data['source_db_id']}")

# ==================== DOWNLOAD SECTION ====================

def create_download_section(clingen_data, myvariant_data, vep_data, classification):
    """Create download section with proper state management."""
    st.subheader("📥 Download Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        if clingen_data:
            clingen_json = json.dumps(clingen_data, indent=2)
            st.download_button(label="📋 ClinGen Data", data=clingen_json, file_name=f"clingen_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json", mime="application/json", key=f"clingen_dl_{classification.extracted_identifier}", help="Download ClinGen Allele Registry data as JSON")
    with col2:
        if myvariant_data:
            myvariant_json = json.dumps(myvariant_data, indent=2)
            st.download_button(label="🔬 MyVariant Data", data=myvariant_json, file_name=f"myvariant_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json", mime="application/json", key=f"myvariant_dl_{classification.extracted_identifier}", help="Download MyVariant.info annotations as JSON")
    with col3:
        if vep_data:
            vep_json = json.dumps(vep_data, indent=2)
            st.download_button(label="🧬 VEP Data", data=vep_json, file_name=f"vep_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json", mime="application/json", key=f"vep_dl_{classification.extracted_identifier}", help="Download Ensembl VEP predictions as JSON")

# ==================== MAIN APPLICATION ====================

def main():
    st.markdown('<h1 class="main-header">🧬 Genetic Variant Analyzer</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### About")
        st.write("""
        This tool analyzes genetic variants using multiple genomic databases and includes an AI assistant for interpretation.
        - **ClinGen Allele Registry**: Canonical allele identifiers
        - **MyVariant.info**: Comprehensive variant annotations
        - **Ensembl VEP**: Variant effect predictions
        - **AI Assistant**: Powered by OpenAI & Google Gemini for summarization and Q&A
        """)
        st.markdown("### Supported Formats")
        st.code("HGVS: NM_002496.3:c.64C>T")
        st.code("RSID: rs369602258")
        
        st.sidebar.markdown("### 📊 Population Filters")
        st.session_state.freq_threshold = st.sidebar.slider(
            "Max Allele Frequency", 
            min_value=0.0, max_value=1.0, value=st.session_state.get('freq_threshold', 1.0), 
            step=0.001, format="%.3f", 
            help="Show populations with allele frequency at or below this value."
        )

    st.markdown('<div class="section-header">Variant Input</div>', unsafe_allow_html=True)
    default_value = getattr(st.session_state, 'example_input', "")
    user_input = st.text_input("Enter a genetic variant (HGVS notation or RSID):", value=default_value, placeholder="e.g., NM_002496.3:c.64C>T or rs369602258", key="variant_input")
    
    if hasattr(st.session_state, 'example_input'):
        delattr(st.session_state, 'example_input')
    
    analyze_button = st.button("🔬 Analyze Variant", type="primary", key="analyze_btn")
    
    should_analyze = analyze_button and user_input
    should_show_results = False
    
    if should_analyze:
        if 'analysis_data' not in st.session_state or st.session_state.get('last_query') != user_input:
            with st.spinner("Analyzing variant..."):
                router = GenomicQueryRouter()
                classification = router.classify_query(user_input)
                
                if not classification.is_genomic:
                    st.error("Invalid input format. Please provide a valid HGVS notation or RSID.")
                    st.stop()
                
                try:
                    start_time = time.time()
                    if classification.query_type == 'rsid':
                        st.info("🔍 RSID detected - querying MyVariant.info and Ensembl VEP directly (skipping ClinGen)")
                        clingen_data = {'CAid': 'N/A (RSID input)', 'rsid': classification.extracted_identifier.replace('rs', ''), 'genomic_hgvs_grch38': None, 'genomic_hgvs_grch37': None, 'myvariant_hg38': None, 'myvariant_hg19': None, 'mane_ensembl': None, 'mane_refseq': None}
                        annotations = get_variant_annotations(clingen_data, classification)
                        if annotations['myvariant_data']:
                            myv_data = annotations['myvariant_data']
                            if isinstance(myv_data, list) and len(myv_data) > 0:
                                myv_data = myv_data[0]; annotations['myvariant_data'] = myv_data
                            if isinstance(myv_data, dict):
                                if myv_data.get('clingen', {}).get('caid'): clingen_data['CAID'] = myv_data['clingen']['caid']
                                hgvs_data = myv_data.get('clinvar', {}).get('hgvs', {})
                                if isinstance(hgvs_data, dict) and hgvs_data.get('coding'):
                                    try:
                                        vep_url = f"https://rest.ensembl.org/vep/human/hgvs/{hgvs_data['coding']}"
                                        vep_headers = {"Content-Type": "application/json", "Accept": "application/json"}
                                        vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                                        if vep_response.ok: annotations['vep_data'] = vep_response.json()
                                    except: pass
                    else:
                        clingen_raw = query_clingen_allele(classification.extracted_identifier)
                        clingen_data = parse_caid_minimal(clingen_raw)
                        annotations = get_variant_annotations(clingen_data, classification)
                    
                    processing_time = time.time() - start_time
                    st.session_state.analysis_data = {'classification': classification, 'clingen_data': clingen_data, 'annotations': annotations, 'processing_time': processing_time}
                    st.session_state.last_query = user_input
                    if 'messages' in st.session_state: del st.session_state.messages
                    should_show_results = True
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}"); st.exception(e); st.stop()
        else:
            should_show_results = True
    
    elif 'analysis_data' in st.session_state and st.session_state.get('last_query'):
        should_show_results = True
    
    if should_show_results and 'analysis_data' in st.session_state:
        analysis_data = st.session_state.analysis_data
        classification, clingen_data, annotations, processing_time = analysis_data['classification'], analysis_data['clingen_data'], analysis_data['annotations'], analysis_data['processing_time']
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Results", key="clear_results"):
                if 'analysis_data' in st.session_state: del st.session_state['analysis_data']
                if 'last_query' in st.session_state: del st.session_state['last_query']
                if 'messages' in st.session_state: del st.session_state['messages']
                st.rerun()
        with col2: st.markdown(f"**Analyzing:** {classification.extracted_identifier}")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: st.write(f"**Detected identifier:** {classification.extracted_identifier}")
        with col2: st.write(f"**Type:** {classification.query_type}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">ClinGen Allele Registry</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**CAid:** {clingen_data.get('CAid', 'N/A')}")
            st.write(f"**RSID:** {clingen_data.get('rsid', 'N/A')}")
        with col2:
            st.write(f"**MANE Ensembl:** {clingen_data.get('mane_ensembl', 'N/A')}")
            st.write(f"**MyVariant ID:** {clingen_data.get('myvariant_hg38', 'N/A')}")
        
        if annotations['errors']:
            for error in annotations['errors']: st.warning(f"⚠️ {error}")
        
        if annotations['myvariant_data'] or annotations['vep_data']:
            st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
            
            result_tabs = ["🧬 VEP Analysis", "🔬 MyVariant Analysis", "🏥 Clinical Data", "🤖 AI Assistant", "📋 Raw Data"]
            tab1, tab2, tab3, tab4, tab5 = st.tabs(result_tabs)
            
            with tab1:
                if annotations['vep_data']: display_vep_analysis(annotations['vep_data'])
                else: st.info("No VEP data available.")
            with tab2:
                if annotations['myvariant_data']: display_comprehensive_myvariant_data(annotations['myvariant_data'])
                else: st.info("No MyVariant data available.")
            with tab3:
                if annotations['myvariant_data']:
                    myvariant_data = annotations['myvariant_data']
                    clinvar_data = myvariant_data.get('clinvar', {})
                    if clinvar_data:
                        st.subheader("ClinVar Clinical Significance")
                        clinical_sig = (clinvar_data.get('clinical_significance') or clinvar_data.get('clnsig') or 'N/A')
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("Clinical Significance", clinical_sig)
                        with col2:
                            if clinvar_data.get('variant_id'): st.metric("ClinVar ID", clinvar_data['variant_id'])
                        with col3:
                            if clinvar_data.get('allele_id'): st.metric("Allele ID", clinvar_data['allele_id'])
                        rcv_data = clinvar_data.get('rcv', [])
                        if rcv_data and isinstance(rcv_data, list):
                            st.subheader("Submission Details")
                            for rcv in rcv_data:
                                if isinstance(rcv, dict):
                                    with st.expander(f"ClinVar Record: {rcv.get('accession', 'N/A')}"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**Review Status:** {rcv.get('review_status', 'N/A')}")
                                            st.write(f"**Last Evaluated:** {rcv.get('last_evaluated', 'N/A')}")
                                            st.write(f"**Number of Submitters:** {rcv.get('number_submitters', 'N/A')}")
                                        with col2:
                                            st.write(f"**Origin:** {rcv.get('origin', 'N/A')}")
                                            conditions = rcv.get('conditions', {})
                                            if isinstance(conditions, dict) and conditions.get('name'): st.write(f"**Associated Condition:** {conditions['name']}")
                    uniprot_data = myvariant_data.get('uniprot', {})
                    if uniprot_data and uniprot_data.get('clinical_significance'):
                        st.subheader("UniProt Clinical Annotation")
                        st.write(f"**Clinical Significance:** {uniprot_data['clinical_significance']}")
                        if uniprot_data.get('source_db_id'): st.write(f"**Source:** {uniprot_data['source_db_id']}")
                    st.subheader("Population Frequency Context")
                    max_freq, freq_source = 0, "N/A"
                    gnomad_exome = myvariant_data.get('gnomad_exome', {})
                    if gnomad_exome and gnomad_exome.get('af', {}).get('af'):
                        exome_freq = gnomad_exome['af']['af']
                        if exome_freq > max_freq: max_freq, freq_source = exome_freq, "gnomAD Exome"
                    gnomad_genome = myvariant_data.get('gnomad_genome', {})
                    if gnomad_genome and gnomad_genome.get('af', {}).get('af'):
                        genome_freq = gnomad_genome['af']['af']
                        if genome_freq > max_freq: max_freq, freq_source = genome_freq, "gnomAD Genome"
                    if max_freq > 0:
                        st.metric(f"Max Population Frequency ({freq_source})", f"{max_freq:.6f}")
                        if max_freq >= 0.01: st.success("🟢 Common variant (≥1%)")
                        elif max_freq >= 0.005: st.warning("🟡 Low frequency variant (0.5-1%)")
                        elif max_freq >= 0.0001: st.info("🔵 Rare variant (0.01-0.5%)")
                        else: st.error("🔴 Very rare variant (<0.01%)")
                    else: st.info("No reliable population frequency data available")
                else: st.info("No clinical data available")
            
            with tab4:
                display_ai_assistant(analysis_data)
            
            with tab5:
                st.subheader("Raw API Responses")
                with st.expander("ClinGen Allele Registry Data", expanded=False): st.json(clingen_data)
                if annotations['myvariant_data']:
                    with st.expander("MyVariant.info Data", expanded=False): st.json(annotations['myvariant_data'])
                if annotations['vep_data']:
                    with st.expander("Ensembl VEP Data", expanded=False): st.json(annotations['vep_data'])
                st.markdown("---")
                create_download_section(clingen_data, annotations['myvariant_data'], annotations['vep_data'], classification)
        
        st.success(f"✅ Analysis completed in {processing_time:.2f} seconds")
        
    elif user_input and not analyze_button:
        router = GenomicQueryRouter()
        classification = router.classify_query(user_input)
        if classification.is_genomic:
            st.success(f"✅ Valid {classification.query_type} format detected: {classification.extracted_identifier}")
        else:
            st.error("❌ Invalid format. Please provide a valid HGVS notation or RSID.")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🧬 <strong>Genetic Variant Analyzer</strong></p>
        <p>Data sources: ClinGen Allele Registry • MyVariant.info • Ensembl VEP • OpenAI • Google Gemini</p>
        <p>For research purposes only • Not for clinical use</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
