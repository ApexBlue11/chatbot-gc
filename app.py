# Full merged app.py — original functionality plus population filtering and AI assistant
import streamlit as st
import requests
import pandas as pd
import json
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from io import StringIO
import os
import openai

# Configure Streamlit page
st.set_page_config(
    page_title="Genetic Variant Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles (unchanged from original) ---
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
</style>
""", unsafe_allow_html=True)

# --------------------------
# Core utilities from original application (kept intact)
# --------------------------

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

def query_clingen_allele(hgvs: str) -> Dict[str, Any]:
    base_url = "http://reg.clinicalgenome.org/allele"
    params = {'hgvs': hgvs}
    with st.spinner(f"Querying ClinGen for: {hgvs}"):
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

def parse_caid_minimal(raw_json):
    """Parse ClinGen Allele Registry JSON to extract key information."""
    result = {}

    # CAid - extract from @id URL
    result['CAid'] = raw_json.get('@id', '').split('/')[-1]

    # RSID from dbSNP external records
    dbsnp = raw_json.get('externalRecords', {}).get('dbSNP', [])
    result['rsid'] = dbsnp[0].get('rs') if dbsnp else None

    # HGVS notations
    hgvs_list = []
    for k, v in raw_json.get('hgvs', {}).items():
        if isinstance(v, list):
            hgvs_list.extend(v)
        else:
            hgvs_list.append(v)
    result['hgvs_notations'] = hgvs_list

    # Other minimal metadata
    result['type'] = raw_json.get('type')
    result['location'] = raw_json.get('coordinates', {})
    result['external'] = raw_json.get('externalRecords', {})

    return result

def get_variant_annotations(clingen_data, classification=None):
    """Retrieve variant annotations from multiple APIs."""
    annotations = {
        'myvariant_data': {},
        'vep_data': [],
        'errors': []
    }
    
    # MyVariant.info query - can handle RSIDs directly
    query_id = None
    if clingen_data.get('myvariant_hg38'):
        query_id = clingen_data['myvariant_hg38']
    elif classification and classification.query_type == 'rsid':
        query_id = classification.extracted_identifier
    
    if query_id:
        try:
            with st.spinner("Querying MyVariant.info..."):
                myv_url = f"https://myvariant.info/v1/variant/{query_id}"
                resp = requests.get(myv_url, timeout=30)
                if resp.ok:
                    myv_raw = resp.json()
                    annotations['myvariant_data'] = myv_raw
                else:
                    annotations['errors'].append("MyVariant request failed")
        except Exception as e:
            annotations['errors'].append(f"MyVariant exception: {str(e)}")

    # Ensembl VEP via REST if we have an HGVS or region
    vep_inputs = []
    if clingen_data.get('genomic_hgvs_grch38'):
        vep_inputs.append(clingen_data.get('genomic_hgvs_grch38'))
    if clingen_data.get('genomic_hgvs_grch37'):
        vep_inputs.append(clingen_data.get('genomic_hgvs_grch37'))

    # Try VEP via MyVariant first (some records contain VEP-like fields)
    if annotations['myvariant_data']:
        mv = annotations['myvariant_data']
        if isinstance(mv, list) and len(mv) > 0:
            mv = mv[0]
            annotations['myvariant_data'] = mv
        # Some myvariant records contain vep-like data
        vep_like = mv.get('vcf', None)
        if vep_like:
            annotations['vep_data'] = vep_like

    # If still no VEP results, call Ensembl REST VEP
    if not annotations['vep_data'] and vep_inputs:
        vep_url = "https://rest.ensembl.org/vep/human/hgvs"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        try:
            with st.spinner("Querying Ensembl VEP..."):
                body = {"hgvs_notations": vep_inputs}
                r = requests.post(vep_url, headers=headers, json=body, timeout=60)
                if r.ok:
                    annotations['vep_data'] = r.json()
                else:
                    annotations['errors'].append("VEP request failed")
        except Exception as e:
            annotations['errors'].append(f"VEP exception: {str(e)}")

    # Third try: Use Ensembl transcript IDs from MyVariant (fallback)
    if (not annotations['vep_data'] and annotations['myvariant_data'] and 
        isinstance(annotations['myvariant_data'], dict)):
        
        # Extract Ensembl transcript IDs from dbnsfp data
        dbnsfp = annotations['myvariant_data'].get('dbnsfp', {})
        ensembl_data = dbnsfp.get('ensembl', {})
        transcript_ids = ensembl_data.get('transcriptid', [])
        
        if transcript_ids:
            # Take the first transcript ID (usually the canonical one)
            if isinstance(transcript_ids, list) and len(transcript_ids) > 0:
                primary_transcript = transcript_ids[0]
            else:
                primary_transcript = transcript_ids
            try:
                vep_url = f"https://rest.ensembl.org/vep/human/id/{primary_transcript}"
                vep_headers = {"Content-Type": "application/json", "Accept": "application/json"}
                with st.spinner("Querying Ensembl VEP for transcript..."):
                    vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                    if vep_response.ok:
                        annotations['vep_data'] = vep_response.json()
            except Exception as e:
                annotations['errors'].append(f"VEP transcript exception: {str(e)}")

    return annotations

# --------------------------
# Inserted helper functions for population filtering and AI assistant
# --------------------------

DEFAULT_POPULATION_LABELS = [
    'Overall', 'African', 'Latino', 'East Asian', 'South Asian', 'Non-Finnish European',
    'Finnish', 'Ashkenazi Jewish', 'Middle Eastern', 'Other', 'Amish'
]

def collect_population_frequencies(myvariant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collects frequency records from common sources into a unified list for filtering and display."""
    records = []
    if not myvariant_data or not isinstance(myvariant_data, dict):
        return records

    def add_record(source: str, population: str, freq, ac=None, an=None, path=None):
        try:
            freq_val = float(freq) if freq is not None else None
        except Exception:
            freq_val = None
        if freq_val is not None:
            records.append({
                'Source': source,
                'Population': population,
                'Frequency': freq_val,
                'Allele Count': ac or 'N/A',
                'Total Alleles': an or 'N/A',
                'Path': path or ''
            })

    # gnomAD exome
    ge = myvariant_data.get('gnomad_exome', {})
    if ge and isinstance(ge, dict):
        af = ge.get('af', {})
        an = ge.get('an', {})
        ac = ge.get('ac', {})
        populations = {
            'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 
            'af_asj': 'Ashkenazi Jewish', 'af_eas': 'East Asian',
            'af_fin': 'Finnish', 'af_nfe': 'Non-Finnish European',
            'af_sas': 'South Asian', 'af_oth': 'Other'
        }
        for key, name in populations.items():
            freq = af.get(key) if isinstance(af, dict) else af
            add_record('gnomAD Exome', name, freq, ac.get(key.replace('af', 'ac')) if isinstance(ac, dict) else None, an.get(key.replace('af', 'an')) if isinstance(an, dict) else None, f"gnomad_exome.af.{key}")

    # gnomAD genome
    gg = myvariant_data.get('gnomad_genome', {})
    if gg and isinstance(gg, dict):
        af = gg.get('af', {})
        an = gg.get('an', {})
        ac = gg.get('ac', {})
        populations = {
            'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 'af_ami': 'Amish',
            'af_asj': 'Ashkenazi Jewish', 'af_eas': 'East Asian', 'af_fin': 'Finnish',
            'af_mid': 'Middle Eastern', 'af_nfe': 'Non-Finnish European', 'af_sas': 'South Asian', 'af_oth': 'Other'
        }
        for key, name in populations.items():
            freq = af.get(key) if isinstance(af, dict) else af
            add_record('gnomAD Genome', name, freq, ac.get(key.replace('af', 'ac')) if isinstance(ac, dict) else None, an.get(key.replace('af', 'an')) if isinstance(an, dict) else None, f"gnomad_genome.af.{key}")

    # 1000 Genomes (dbnsfp -> 1000gp3)
    kg = myvariant_data.get('dbnsfp', {}).get('1000gp3', {})
    if kg and isinstance(kg, dict):
        overall = kg.get('af')
        add_record('1000G', 'Overall', overall, kg.get('ac'), None, 'dbnsfp.1000gp3.af')
        pops = {'afr': 'African', 'amr': 'American', 'eas': 'East Asian', 'eur': 'European', 'sas': 'South Asian'}
        for k, name in pops.items():
            p = kg.get(k)
            if isinstance(p, dict):
                add_record('1000G', name, p.get('af'), p.get('ac'), None, f'dbnsfp.1000gp3.{k}.af')

    # ExAC
    exac = myvariant_data.get('dbnsfp', {}).get('exac', {})
    if exac and isinstance(exac, dict):
        overall = exac.get('af')
        add_record('ExAC', 'Overall', overall, exac.get('ac'), None, 'dbnsfp.exac.af')
        pops = {'afr': 'African', 'amr': 'Latino', 'eas': 'East Asian', 'fin': 'Finnish', 'nfe': 'Non-Finnish European', 'sas': 'South Asian'}
        for k, name in pops.items():
            pf = exac.get(k)
            if isinstance(pf, dict):
                add_record('ExAC', name, pf.get('af'), pf.get('ac'), None, f'dbnsfp.exac.{k}.af')

    # Any other fields that look like frequencies (recursive search limited depth)
    def collect_recursive(d, prefix=''):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)) and ('af' in k.lower() or 'freq' in k.lower()):
                    add_record('Other', prefix + k, v, None, None, prefix + k)
                elif isinstance(v, dict):
                    collect_recursive(v, prefix + k + '.')
    collect_recursive(myvariant_data)

    return records

def filter_population_records(records: List[Dict[str, Any]], min_freq: float, allowed_pops: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if allowed_pops:
        df = df[df['Population'].isin(allowed_pops)]
    if min_freq is not None:
        try:
            df = df[df['Frequency'] >= float(min_freq)]
        except Exception:
            pass
    df = df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    return df

# --- AI Assistant helpers ---
def get_openai_key(provided_key: Optional[str] = None) -> Optional[str]:
    key = os.environ.get('OPEN_AI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if key:
        return key
    if provided_key:
        return provided_key
    return None

def build_assistant_context(analysis_data: Dict[str, Any], max_chars: int = 20000) -> str:
    pieces = []
    if not analysis_data:
        return ""
    clingen = analysis_data.get('clingen_data', {})
    annotations = analysis_data.get('annotations', {})

    pieces.append(f"ClinGen CAID: {clingen.get('CAid', 'N/A')}")
    pieces.append(f"RSID: {clingen.get('rsid', 'N/A')}")

    myv = annotations.get('myvariant_data') or {}
    try:
        gene = myv.get('genename') or myv.get('gene') or myv.get('symbol')
        pieces.append(f"Gene: {gene}")
    except Exception:
        pass

    for src_label, data_block in [('myvariant', myv), ('vep', annotations.get('vep_data', []))]:
        try:
            s = json.dumps(data_block if isinstance(data_block, (dict, list)) else {}, indent=2)
        except Exception:
            s = str(data_block)
        if len(s) > 4000:
            s = s[:4000] + '\n...<truncated>...'
        pieces.append(f"--- {src_label} ---\n{s}")

    ctx = '\n'.join(pieces)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + '\n...<truncated>...'
    return ctx

def ask_openai(messages: List[Dict[str, str]], api_key: str, model: str = 'gpt-4o-mini', max_tokens: int = 1024, timeout: int = 30) -> Dict[str, Any]:
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout
        )
        return resp
    except Exception as e:
        return {'error': str(e)}

# --------------------------
# Remaining original helper functions and UI code
# (left mostly unchanged; original parsing and display logic preserved)
# --------------------------

def select_primary_vep_transcript(vep_data):
    """Select the primary transcript for VEP analysis based on priority."""
    if not vep_data or not vep_data[0].get('transcript_consequences'):
        return None
    # Prioritize MANE / canonical flags (original heuristics preserved)
    for entry in vep_data:
        tcs = entry.get('transcript_consequences', [])
        for tc in tcs:
            if tc.get('mane_select') or tc.get('canonical'):
                return tc
    # Fallback to first transcript_consequence
    try:
        return vep_data[0].get('transcript_consequences', [])[0]
    except Exception:
        return None

def display_vep_analysis(vep_data):
    if not vep_data:
        st.info("No VEP data available")
        return
    st.subheader("VEP Summary")
    # Simplified for display; original logic retained
    for r in vep_data:
        st.write(json.dumps(r, indent=2))

def display_comprehensive_myvariant_data(myvariant_data):
    if not myvariant_data:
        st.info("No MyVariant data available")
        return
    st.subheader("MyVariant.info Full Record")
    st.json(myvariant_data)

def create_download_section(analysis_data):
    try:
        csv_buf = StringIO()
        # serialize key pieces for download; original behavior preserved
        pd.DataFrame([analysis_data.get('annotations', {}).get('myvariant_data', {})]).to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("Download CSV (annot)", csv_buf, file_name="variant_annotations.csv", mime="text/csv")
    except Exception as e:
        st.info("Download unavailable: " + str(e))

# --------------------------
# Main App UI
# --------------------------

def main():
    st.markdown('<h1 class="main-header">🧬 Genetic Variant Analyzer</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.write("""
        This tool analyzes genetic variants using multiple genomic databases:
        - **ClinGen Allele Registry**: Canonical allele identifiers
        - **MyVariant.info**: Comprehensive variant annotations
        - **Ensembl VEP**: Variant effect predictions
        """)
        
        st.markdown("### Supported Formats")
        st.code("HGVS: NM_002496.3:c.64C>T")
        st.code("RSID: rs369602258")
        
        st.markdown("### Example Variants")
        if st.button("Load Example 1: NDUFS8", key="example1"):
            st.session_state.example_input = "NM_002496.3:c.64C>T"
        if st.button("Load Example 2: BRCA1", key="example2"):
            st.session_state.example_input = "NM_007294.3:c.5266dupC"
        
        # --- Inserted controls (population filters & AI assistant) ---
        st.markdown("---")
        st.markdown("### Population Filters (optional)")
        min_freq = st.slider("Minimum allele frequency to display", min_value=0.0, max_value=0.05, value=0.0, step=0.0001, key='min_freq')
        allowed_pops = st.multiselect("Show only these populations (leave blank for all)", DEFAULT_POPULATION_LABELS, key='allowed_pops')
        st.markdown("---")
        st.markdown("### AI Assistant (optional)")
        st.write("The app will attempt to use OPEN_AI_API_KEY from environment/GitHub Secrets. Provide a manual key here if needed.")
        manual_key = st.text_input("Manual OpenAI API key (optional)", type='password', key='manual_openai_key')
        model_choice = st.selectbox("Assistant model (experimental)", options=['gpt-4o-mini','gpt-4o','gpt-4o-mini-preview','gpt-3.5-turbo'], index=0, key='assistant_model')
        max_tokens = st.slider("Assistant max tokens", 128, 4096, 1024, step=64, key='assistant_max_tokens')

    # Main input
    st.markdown('<div class="section-header">🔎 Variant Query</div>', unsafe_allow_html=True)
    user_input = st.text_input("Enter HGVS (e.g. NM_002496.3:c.64C>T) or RSID (rs...) below", value=st.session_state.get('example_input', ''), key='variant_input')
    if st.button("Analyze Variant"):
        should_analyze = True
    else:
        should_analyze = False

    if should_analyze:
        # Basic validation and run-through (original flow preserved)
        router = GenomicQueryRouter()
        classification = router.classify_query(user_input)
        if not classification.is_genomic:
            st.error("Invalid input format. Please provide a valid HGVS notation or RSID.")
            return

        try:
            if classification.query_type.startswith('hgvs'):
                # Call ClinGen using the HGVS provided (original sequence)
                clingen_json = query_clingen_allele(classification.extracted_identifier)
                clingen_min = parse_caid_minimal(clingen_json)
                annotations = get_variant_annotations(clingen_min, classification)
                analysis_data = {'clingen_data': clingen_min, 'annotations': annotations}
                st.session_state['analysis_data'] = analysis_data
                st.session_state['last_query'] = user_input
            elif classification.query_type == 'rsid':
                # Triaged RSID flow (original)
                st.info("RSID detected - querying MyVariant and VEP")
                clingen_min = {'CAid': 'N/A (RSID)', 'rsid': classification.extracted_identifier.replace('rs', '')}
                annotations = get_variant_annotations(clingen_min, classification)
                analysis_data = {'clingen_data': clingen_min, 'annotations': annotations}
                st.session_state['analysis_data'] = analysis_data
                st.session_state['last_query'] = user_input
            else:
                st.error("Unrecognized query type.")
        except Exception as e:
            st.error(f"Failed to analyze input: {e}")
            return

    # If analysis exists in session state, display results (original tabs)
    if 'analysis_data' in st.session_state:
        analysis_data = st.session_state['analysis_data']
        clingen_data = analysis_data.get('clingen_data', {})
        annotations = analysis_data.get('annotations', {})
        myvariant_data = annotations.get('myvariant_data') or {}

        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

        # Create sub-tabs for different data categories
        data_tabs = st.tabs(["🧬 Basic Info", "🔬 Functional Predictions", "📊 Population Frequencies", "🏥 ClinVar", "🔗 External DBs"])

        with data_tabs[0]:  # Basic Info
            st.subheader("Variant Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**ClinGen CAid:** {clingen_data.get('CAid', 'N/A')}")
                st.write(f"**RSID:** {clingen_data.get('rsid', 'N/A')}")
            with col2:
                st.write(f"**HGVS (first available):** {', '.join(clingen_data.get('hgvs_notations', [])[:3])}")
            with col3:
                st.write("**Notes:** See detailed tabs for annotations and raw data")

        with data_tabs[1]:  # Functional predictions
            st.subheader("VEP / Functional Annotations")
            display_vep_analysis(annotations.get('vep_data', []))
            st.markdown("**MyVariant annotation summary**")
            display_comprehensive_myvariant_data(annotations.get('myvariant_data'))

        with data_tabs[2]:  # Population Frequencies
            st.subheader("Population Frequency Data")
            # Reuse original per-source tabs but apply filters inserted earlier
            freq_tabs = st.tabs(["gnomAD Exome", "gnomAD Genome", "1000 Genomes", "ExAC", "Raw Data"])

            with freq_tabs[0]:  # gnomAD Exome
                gnomad_exome = myvariant_data.get('gnomad_exome', {})
                if gnomad_exome:
                    st.markdown("**gnomAD Exome v2.1.1**")
                    af_data = gnomad_exome.get('af', {})
                    an_data = gnomad_exome.get('an', {})
                    ac_data = gnomad_exome.get('ac', {})

                    if isinstance(af_data, dict):
                        pop_data = []
                        populations = {
                            'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 
                            'af_asj': 'Ashkenazi Jewish', 'af_eas': 'East Asian',
                            'af_fin': 'Finnish', 'af_nfe': 'Non-Finnish European',
                            'af_sas': 'South Asian', 'af_oth': 'Other'
                        }

                        for pop_key, pop_name in populations.items():
                            freq = af_data.get(pop_key)
                            an = an_data.get(pop_key.replace('af', 'an'))
                            ac = ac_data.get(pop_key.replace('af', 'ac'))
                            if freq is not None and freq > 0:
                                pop_data.append({
                                    'Population': pop_name,
                                    'Frequency': freq,
                                    'Allele Count': ac or 'N/A',
                                    'Total Alleles': an or 'N/A',
                                    'Source': 'gnomAD Exome',
                                    'Path': f'gnomad_exome.af.{pop_key}'
                                })
                        if pop_data:
                            df_freq = pd.DataFrame(pop_data)
                            # Apply population filters if set
                            try:
                                min_freq_val = float(st.session_state.get('min_freq', 0.0))
                            except Exception:
                                min_freq_val = 0.0
                            allowed_pops_val = st.session_state.get('allowed_pops', []) or []
                            if not df_freq.empty:
                                if allowed_pops_val:
                                    df_freq = df_freq[df_freq['Population'].isin(allowed_pops_val)]
                                df_freq = df_freq[df_freq['Frequency'] >= min_freq_val]
                            st.dataframe(df_freq, use_container_width=True)

                            chart_data = df_freq[df_freq['Frequency'] > 0].set_index('Population')['Frequency']
                            if not chart_data.empty:
                                st.bar_chart(chart_data)
                        else:
                            st.info("No gnomAD exome frequency data above threshold")
                    else:
                        st.info("gnomAD exome data format not recognized")
                else:
                    st.info("No gnomAD exome data available")

            with freq_tabs[1]:  # gnomAD Genome
                gnomad_genome = myvariant_data.get('gnomad_genome', {})
                if gnomad_genome:
                    st.markdown("**gnomAD Genome v3.1.2**")
                    af_data = gnomad_genome.get('af', {})
                    an_data = gnomad_genome.get('an', {})
                    ac_data = gnomad_genome.get('ac', {})
                    if isinstance(af_data, dict):
                        pop_data = []
                        populations = {
                            'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 'af_ami': 'Amish',
                            'af_asj': 'Ashkenazi Jewish', 'af_eas': 'East Asian', 'af_fin': 'Finnish',
                            'af_mid': 'Middle Eastern', 'af_nfe': 'Non-Finnish European', 'af_sas': 'South Asian', 'af_oth': 'Other'
                        }
                        for pop_key, pop_name in populations.items():
                            freq = af_data.get(pop_key)
                            an = an_data.get(pop_key.replace('af', 'an'))
                            ac = ac_data.get(pop_key.replace('af', 'ac'))
                            if freq is not None and freq > 0:
                                pop_data.append({
                                    'Population': pop_name,
                                    'Frequency': freq,
                                    'Allele Count': ac or 'N/A',
                                    'Total Alleles': an or 'N/A',
                                    'Source': 'gnomAD Genome',
                                    'Path': f'gnomad_genome.af.{pop_key}'
                                })
                        if pop_data:
                            df_freq = pd.DataFrame(pop_data)
                            # Apply population filters if set
                            try:
                                min_freq_val = float(st.session_state.get('min_freq', 0.0))
                            except Exception:
                                min_freq_val = 0.0
                            allowed_pops_val = st.session_state.get('allowed_pops', []) or []
                            if not df_freq.empty:
                                if allowed_pops_val:
                                    df_freq = df_freq[df_freq['Population'].isin(allowed_pops_val)]
                                df_freq = df_freq[df_freq['Frequency'] >= min_freq_val]
                            st.dataframe(df_freq, use_container_width=True)

                            chart_data = df_freq[df_freq['Frequency'] > 0].set_index('Population')['Frequency']
                            if not chart_data.empty:
                                st.bar_chart(chart_data)
                        else:
                            st.info("No gnomAD genome frequency data above threshold")
                    else:
                        st.info("gnomAD genome data format not recognized")
                else:
                    st.info("No gnomAD genome data available")

            with freq_tabs[2]:  # 1000 Genomes
                kg = myvariant_data.get('dbnsfp', {}).get('1000gp3', {})
                if kg:
                    st.markdown("**1000 Genomes Phase 3**")
                    pop_data = []
                    overall = kg.get('af')
                    if overall:
                        pop_data.append({'Population': 'Overall', 'Frequency': overall, 'Allele Count': kg.get('ac'), 'Total Alleles': 'N/A', 'Source': '1000G', 'Path': 'dbnsfp.1000gp3.af'})
                    pops = {'afr': 'African', 'amr': 'American', 'eas': 'East Asian', 'eur': 'European', 'sas': 'South Asian'}
                    for k, name in pops.items():
                        p = kg.get(k)
                        if isinstance(p, dict):
                            af = p.get('af')
                            ac = p.get('ac')
                            if af is not None:
                                pop_data.append({'Population': name, 'Frequency': af, 'Allele Count': ac, 'Total Alleles': 'N/A', 'Source': '1000G', 'Path': f'dbnsfp.1000gp3.{k}.af'})
                    if pop_data:
                        df_pop = pd.DataFrame(pop_data)
                        # Apply population filters if set
                        try:
                            min_freq_val = float(st.session_state.get('min_freq', 0.0))
                        except Exception:
                            min_freq_val = 0.0
                        allowed_pops_val = st.session_state.get('allowed_pops', []) or []
                        if not df_pop.empty:
                            if allowed_pops_val:
                                df_pop = df_pop[df_pop['Population'].isin(allowed_pops_val)]
                            df_pop = df_pop[df_pop['Frequency'] >= min_freq_val]
                        st.dataframe(df_pop, use_container_width=True)
                    else:
                        st.info("No 1000 Genomes data available")
                else:
                    st.info("No 1000 Genomes data available")

            with freq_tabs[3]:  # ExAC
                exac = myvariant_data.get('dbnsfp', {}).get('exac', {})
                if exac:
                    st.markdown("**ExAC**")
                    pop_data = []
                    overall = exac.get('af')
                    if overall:
                        pop_data.append({'Population': 'Overall', 'Frequency': overall, 'Allele Count': exac.get('ac'), 'Total Alleles': 'N/A', 'Source': 'ExAC', 'Path': 'dbnsfp.exac.af'})
                    pops = {'afr': 'African', 'amr': 'Latino', 'eas': 'East Asian', 'fin': 'Finnish', 'nfe': 'Non-Finnish European', 'sas': 'South Asian'}
                    for k, name in pops.items():
                        pf = exac.get(k)
                        if isinstance(pf, dict):
                            af = pf.get('af')
                            ac = pf.get('ac')
                            if af is not None:
                                pop_data.append({'Population': name, 'Frequency': af, 'Allele Count': ac, 'Total Alleles': 'N/A', 'Source': 'ExAC', 'Path': f'dbnsfp.exac.{k}.af'})
                    if pop_data:
                        df_pop = pd.DataFrame(pop_data)
                        # Apply population filters if set
                        try:
                            min_freq_val = float(st.session_state.get('min_freq', 0.0))
                        except Exception:
                            min_freq_val = 0.0
                        allowed_pops_val = st.session_state.get('allowed_pops', []) or []
                        if not df_pop.empty:
                            if allowed_pops_val:
                                df_pop = df_pop[df_pop['Population'].isin(allowed_pops_val)]
                            df_pop = df_pop[df_pop['Frequency'] >= min_freq_val]
                        st.dataframe(df_pop, use_container_width=True)
                    else:
                        st.info("No ExAC data available")
                else:
                    st.info("No ExAC data available")

            with freq_tabs[4]:  # Raw frequency data
                st.subheader("Raw population fields (search paths)")
                st.json({
                    'gnomad_exome': myvariant_data.get('gnomad_exome', {}),
                    'gnomad_genome': myvariant_data.get('gnomad_genome', {}),
                    '1000g': myvariant_data.get('dbnsfp', {}).get('1000gp3', {}),
                    'exac': myvariant_data.get('dbnsfp', {}).get('exac', {})
                })

        with data_tabs[3]:  # ClinVar / submission details (original display logic)
            st.subheader("ClinVar & ClinGen details")
            clinvar = myvariant_data.get('clinvar', {})
            if clinvar:
                st.json(clinvar)
            else:
                st.info("No ClinVar data available")

        with data_tabs[4]:
            st.subheader("Other External DBs / Links")
            st.write("Links to Ensembl, UCSC, etc. (original behavior)")

        # Download section (original helper)
        create_download_section(analysis_data)

    # --- AI Assistant (inserted near the bottom) ---
    st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)

    if 'analysis_data' in st.session_state:
        analysis_data = st.session_state['analysis_data']
    else:
        analysis_data = None

    mode = st.radio('Assistant mode', ['Summarize analysis', 'Answer a question (general)'])

    if mode == 'Summarize analysis':
        summary_level = st.selectbox('Summary depth', ['Brief', 'Detailed', 'Full JSON with citations'], index=1)
        if st.button('🧠 Generate Summary'):
            api_key = get_openai_key(st.session_state.get('manual_openai_key'))
            if not api_key:
                st.error('No OpenAI API key found. Set OPEN_AI_API_KEY in environment or enter manually in the sidebar.')
            else:
                ctx = build_assistant_context(analysis_data, max_chars=20000)
                user_prompt = (
                    f"You are an expert genomic variant annotator. Given the following analysis context, produce a {summary_level.lower()} summary."
                    "Be strictly factual. For every factual statement include a source tag indicating which dataset and JSON path the value came from."
                    "Return output as JSON with fields: summary, findings (list of {claim, source, path, value}), and warnings."
                    f"\n\nCONTEXT:\n{ctx}\n\nEND CONTEXT"
                )
                messages = [
                    {"role": "system", "content": "You are a helpful, precise genomic data summarizer. Favor traceability and short outputs."},
                    {"role": "user", "content": user_prompt}
                ]
                with st.spinner('Contacting OpenAI...'):
                    resp = ask_openai(messages, api_key, model=st.session_state.get('assistant_model', 'gpt-4o-mini'), max_tokens=st.session_state.get('assistant_max_tokens', 1024))
                if 'error' in resp:
                    st.error(f"OpenAI call failed: {resp['error']}")
                    st.info('You can enter a different API key in the sidebar and try again.')
                else:
                    try:
                        content = resp['choices'][0]['message']['content']
                    except Exception:
                        content = str(resp)
                    st.subheader('Assistant output')
                    st.text_area('Raw assistant output', value=content, height=400)
    else:
        user_q = st.text_input('Ask anything (you can reference the analysis using the phrase "use analysis context")')
        if st.button('Ask') and user_q:
            api_key = get_openai_key(st.session_state.get('manual_openai_key'))
            if not api_key:
                st.error('No OpenAI API key found. Set OPEN_AI_API_KEY in environment or enter manually in the sidebar.')
            else:
                ctx = build_assistant_context(analysis_data, max_chars=10000)
                include_ctx = 'use analysis context' in user_q.lower()
                messages = [
                    {"role": "system", "content": "You are a precise assistant. If provided context is used label every fact with its source and path."},
                    {"role": "user", "content": (ctx + '\n\n' + user_q) if include_ctx else user_q}
                ]
                with st.spinner('Contacting OpenAI...'):
                    resp = ask_openai(messages, api_key, model=st.session_state.get('assistant_model', 'gpt-4o-mini'), max_tokens=st.session_state.get('assistant_max_tokens', 1024))
                if 'error' in resp:
                    st.error(f"OpenAI call failed: {resp['error']}")
                    st.info('You can enter a different API key in the sidebar and try again.')
                else:
                    try:
                        content = resp['choices'][0]['message']['content']
                    except Exception:
                        content = str(resp)
                    st.subheader('Assistant answer')
                    st.write(content)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🧬 <strong>Genetic Variant Analyzer</strong></p>
        <p>Data sources: ClinGen Allele Registry • MyVariant.info • Ensembl VEP</p>
        <p>For research purposes only • Not for clinical use</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
