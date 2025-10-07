import streamlit as st
import requests
import pandas as pd
import json
import time
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🧬 Genetic Variant Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UTILITY CLASSES (from original code)
# ============================================================================

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

class PopulationFilterConfig:
    """Filter redundant population frequencies"""
    
    def __init__(self, primary_pops=['sas'], secondary_pops=[], include_global=True):
        self.PRIMARY_POPULATIONS = primary_pops
        self.SECONDARY_POPULATIONS = secondary_pops
        self.INCLUDE_GLOBAL = include_global
        
    def should_include_field(self, field_name: str) -> bool:
        field_lower = field_name.lower()
        
        # Check if population frequency field
        is_pop_freq = False
        freq_prefixes = [
            'gnomad_genome_af_af_', 'gnomad_genome_ac_ac_',
            'gnomad_genome_an_an_', 'gnomad_genome_hom_hom_',
            'gnomad_exome_af_af_', 'gnomad_exome_ac_ac_',
            'gnomad_exome_an_an_', 'gnomad_exome_hom_hom_'
        ]
        
        for prefix in freq_prefixes:
            if field_lower.startswith(prefix):
                is_pop_freq = True
                break
        
        if not is_pop_freq:
            return True
        
        # Keep global frequencies
        if self.INCLUDE_GLOBAL:
            if (field_name.endswith('_af_af') or field_name.endswith('_ac_ac') or 
                field_name.endswith('_an_an') or field_name.endswith('_hom_hom')):
                return True
            if (field_name.endswith('_af_xx') or field_name.endswith('_af_xy') or
                field_name.endswith('_ac_xx') or field_name.endswith('_ac_xy')):
                return True
        
        # Check primary/secondary populations
        for pop in self.PRIMARY_POPULATIONS + self.SECONDARY_POPULATIONS:
            if f'_{pop}_' in field_lower or f'_{pop}' in field_lower.split('_')[-1]:
                return True
        
        return False
    
    def get_category_for_field(self, field_name: str) -> str:
        field_lower = field_name.lower()
        
        if field_name in ['rsid', 'variant_id', '_id', 'chrom', 'pos', 'ref', 'alt']:
            return "VARIANT_IDENTIFICATION"
        if any(x in field_lower for x in ['gene', 'transcript', 'feature']):
            return "GENE_AND_TRANSCRIPT"
        if 'hgvs' in field_lower:
            return "HGVS_NOMENCLATURE"
        if any(x in field_lower for x in ['consequence', 'impact', 'exon']):
            return "VARIANT_CONSEQUENCES"
        if 'clinvar' in field_lower:
            return "CLINICAL_SIGNIFICANCE"
        if any(x in field_lower for x in ['sift', 'polyphen', 'cadd', 'revel']):
            return "FUNCTIONAL_PREDICTIONS"
        if any(x in field_lower for x in ['_af', '_ac', '_an', 'freq']):
            return "POPULATION_FREQUENCIES"
        
        return "ADDITIONAL_ANNOTATIONS"

# ============================================================================
# API FUNCTIONS (ClinGen, VEP, MyVariant)
# ============================================================================

def query_clingen_allele(hgvs: str) -> Dict[str, Any]:
    base_url = "http://reg.clinicalgenome.org/allele"
    params = {'hgvs': hgvs}
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def parse_caid_minimal(raw_json):
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

# ============================================================================
# PARSING UTILITIES (simplified from original)
# ============================================================================

_CONSEQ_RANK = {
    "transcript_ablation": 0, "splice_acceptor_variant": 1,
    "splice_donor_variant": 1, "stop_gained": 2, "frameshift_variant": 2,
    "stop_lost": 2, "start_lost": 3, "missense_variant": 6,
    "synonymous_variant": 11, "intron_variant": 14
}

def _consequence_score(term: str) -> int:
    return _CONSEQ_RANK.get(term, 999)

def _normalize_input(x: Any) -> List:
    if x is None:
        return []
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return []
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return []

def _safe_get(d: dict, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

def _pick_best_transcript(transcripts: List[dict]):
    if not transcripts:
        return None
    
    for t in transcripts:
        if (t.get("canonical") or t.get("mane_select")):
            return t
    
    best_transcript = None
    best_score = 9999
    
    for t in transcripts:
        consequence_terms = t.get("consequence_terms") or []
        if isinstance(consequence_terms, str):
            consequence_terms = [consequence_terms]
        
        severity_score = min((_consequence_score(term) for term in consequence_terms), default=999)
        
        if severity_score < best_score:
            best_score = severity_score
            best_transcript = t
    
    return best_transcript or transcripts[0]

def parse_vep_list(vep_input: Any) -> pd.DataFrame:
    records = _normalize_input(vep_input)
    out_rows = []
    
    for rec in records:
        base_info = {
            "source": "vep",
            "input": rec.get("input") or rec.get("id"),
            "most_severe_consequence": rec.get("most_severe_consequence"),
            "seq_region": rec.get("seq_region_name"),
            "start": rec.get("start"),
            "end": rec.get("end")
        }
        
        transcript_consequences = rec.get("transcript_consequences") or []
        best_transcript = _pick_best_transcript(transcript_consequences)
        
        if best_transcript:
            base_info.update({
                "gene_symbol": best_transcript.get("gene_symbol"),
                "transcript_id": best_transcript.get("transcript_id"),
                "consequence_terms": best_transcript.get("consequence_terms"),
                "hgvsc": best_transcript.get("hgvsc"),
                "hgvsp": best_transcript.get("hgvsp"),
                "sift_prediction": best_transcript.get("sift_prediction"),
                "sift_score": best_transcript.get("sift_score"),
                "polyphen_prediction": best_transcript.get("polyphen_prediction"),
                "polyphen_score": best_transcript.get("polyphen_score"),
            })
        
        out_rows.append(base_info)
    
    return pd.DataFrame(out_rows)

def parse_myv_list(myv_input: Any) -> pd.DataFrame:
    records = _normalize_input(myv_input)
    out_rows = []
    
    for rec in records:
        row = {
            "source": "myvariant",
            "_id": rec.get("_id"),
            "rsid": rec.get("rsid") or _safe_get(rec, ["dbsnp.rsid"]),
            "chrom": _safe_get(rec, ["hg38.chrom", "chrom"]),
            "pos": _safe_get(rec, ["hg38.start", "pos"]),
        }
        
        # ClinVar
        clinvar_block = rec.get("clinvar") or {}
        if isinstance(clinvar_block, dict):
            row["clinvar_clinsig"] = _safe_get(clinvar_block, ["clinsig"])
            row["clinvar_review_status"] = _safe_get(clinvar_block, ["review_status"])
            
            rcv_data = _safe_get(clinvar_block, ["rcv"])
            if isinstance(rcv_data, list):
                rcv_accessions = []
                for item in rcv_data:
                    if isinstance(item, dict):
                        acc = item.get("accession")
                        if acc:
                            rcv_accessions.append(acc)
                row["clinvar_rcv"] = rcv_accessions if rcv_accessions else None
                row["clinvar_rcv_count"] = len(rcv_accessions)
        
        # dbNSFP predictions
        dbnsfp_block = rec.get("dbnsfp") or {}
        if isinstance(dbnsfp_block, dict):
            row["cadd_phred"] = _safe_get(dbnsfp_block, ["cadd.phred"])
            row["sift_score"] = _safe_get(dbnsfp_block, ["sift.score"])
            row["sift_pred"] = _safe_get(dbnsfp_block, ["sift.pred"])
            row["polyphen2_hdiv_score"] = _safe_get(dbnsfp_block, ["polyphen2.hdiv.score"])
            row["polyphen2_hdiv_pred"] = _safe_get(dbnsfp_block, ["polyphen2.hdiv.pred"])
            row["revel_score"] = _safe_get(dbnsfp_block, ["revel.score"])
        
        out_rows.append(row)
    
    return pd.DataFrame(out_rows)

# ============================================================================
# OPENAI INTEGRATION
# ============================================================================

class OpenAIVariantSummarizer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", pop_filter: PopulationFilterConfig = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.pop_filter = pop_filter or PopulationFilterConfig()
    
    def format_dataframe_for_prompt(self, df: pd.DataFrame, source_name: str) -> str:
        if df.empty:
            return f"{source_name}: No data available"
        
        data_dict = df.iloc[0].to_dict()
        
        # Filter fields
        categorized_fields = {}
        for field_name, field_value in data_dict.items():
            if pd.isna(field_value) or field_value == '' or field_value == []:
                continue
            
            if not self.pop_filter.should_include_field(field_name):
                continue
            
            category = self.pop_filter.get_category_for_field(field_name)
            if category not in categorized_fields:
                categorized_fields[category] = {}
            categorized_fields[category][field_name] = field_value
        
        # Format output
        formatted_sections = [f"=== {source_name.upper()} ANNOTATION DATA ===\n"]
        
        category_order = [
            "VARIANT_IDENTIFICATION", "GENE_AND_TRANSCRIPT", "HGVS_NOMENCLATURE",
            "VARIANT_CONSEQUENCES", "CLINICAL_SIGNIFICANCE", "FUNCTIONAL_PREDICTIONS",
            "POPULATION_FREQUENCIES"
        ]
        
        for category in category_order:
            if category not in categorized_fields:
                continue
            
            fields = categorized_fields[category]
            if not fields:
                continue
            
            formatted_sections.append(f"\n{category.replace('_', ' ')}:")
            
            for field_name in sorted(fields.keys()):
                field_value = fields[field_name]
                
                if isinstance(field_value, list):
                    value_str = ', '.join(str(x) for x in field_value[:3])
                    if len(field_value) > 3:
                        value_str += f" ... ({len(field_value)} total)"
                elif isinstance(field_value, float):
                    if field_value < 0.0001:
                        value_str = f"{field_value:.2e}"
                    elif field_value < 0.01:
                        value_str = f"{field_value:.6f}"
                    else:
                        value_str = f"{field_value:.4f}"
                else:
                    value_str = str(field_value)
                    if len(value_str) > 150:
                        value_str = value_str[:150] + "..."
                
                formatted_sections.append(f"  • {field_name}: {value_str}")
        
        return "\n".join(formatted_sections)
    
    def create_clinical_interpretation_prompt(self, data: str, source_name: str) -> str:
        interpretation_rules = """
CRITICAL INTERPRETATION RULES:

1. **Prediction Score Interpretation:**
   - SIFT: 'T' = Tolerated (BENIGN), 'D' = Deleterious (DAMAGING)
   - SIFT score: <0.05 = deleterious, ≥ 0.05 = tolerated
   - PolyPhen: 'B' = Benign, 'P' = Possibly damaging, 'D' = Probably damaging
   - CADD: >20 is damaging, >30 is highly damaging

2. **ClinVar Data Handling:**
   - If clinvar_rcv_count > 0, there ARE ClinVar records
   - Cite the actual RCV count and clinical significance

3. **Population Frequency Citation:**
   - ALWAYS cite numeric values, not just "low" or "high"
   - Format: "South Asian AF = 0.000123 [gnomad_exome_af_af_sas]"

4. **Citation Format:**
   - Every claim must cite the source field: "SIFT score of 0.42 [sift_score]"
"""
        
        prompt = f"""You are a clinical geneticist analyzing {source_name} data. Follow interpretation rules strictly.

{interpretation_rules}

{source_name.upper()} DATA:
{data}

PROVIDE CLINICAL ASSESSMENT:

1. **Pathogenicity Assessment** (2-3 sentences):
   - Cite EXACT numeric values with field names in brackets
   - Correctly interpret prediction codes

2. **Key Concerns** (bullet points with citations):
   - Population frequency with actual AF values
   - Conflicting predictions if any

3. **Functional Impact** (2-3 sentences):
   - Amino acid change if available
   - Consequence type

4. **Clinical Recommendation** (1-2 sentences):
   - Classification suggestion based on evidence

**Remember:** Every claim MUST cite the source field in brackets like [field_name].
"""
        
        return prompt
    
    def generate_individual_summary(self, df: pd.DataFrame, source_name: str) -> str:
        if df.empty:
            return f"No {source_name} data available."
        
        try:
            formatted_data = self.format_dataframe_for_prompt(df, source_name)
            prompt = self.create_clinical_interpretation_prompt(formatted_data, source_name)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical geneticist providing evidence-based variant interpretation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            summary = response.choices[0].message.content
            
            # Add disclaimer
            if "disclaimer" not in summary.lower():
                summary += "\n\n⚠️ **DISCLAIMER**: This is an automated AI interpretation for research purposes only. Not medical advice."
            
            return summary
            
        except Exception as e:
            return f"Error generating {source_name} summary: {str(e)}"
    
    def generate_combined_summary(self, vep_df: pd.DataFrame, myvariant_df: pd.DataFrame) -> Dict[str, str]:
        return {
            'vep': self.generate_individual_summary(vep_df, "VEP"),
            'clinvar': self.generate_individual_summary(myvariant_df, "MyVariant")
        }

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("🧬 Genetic Variant Analysis System")
    st.markdown("Analyze genetic variants using HGVS notation or RSIDs with AI-powered clinical interpretation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input("OpenAI API Key", type="password", 
                                help="Enter your OpenAI API key")
        
        model_choice = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            help="gpt-4o-mini is fastest and cheapest"
        )
        
        st.subheader("Population Filtering")
        primary_pops = st.multiselect(
            "Primary Populations",
            ['sas', 'afr', 'eas', 'nfe', 'amr', 'fin', 'asj', 'mid'],
            default=['sas'],
            help="Population codes: sas=South Asian, afr=African, nfe=European, etc."
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool queries:
        - **ClinGen** Allele Registry
        - **Ensembl VEP** annotations
        - **MyVariant.info** (ClinVar, dbNSFP)
        
        Then uses OpenAI to generate clinical interpretations.
        """)
    
    # Main input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter HGVS notation or RSID",
            placeholder="Example: NM_002496.3:c.64C>T or rs369602258",
            help="Supports transcript (NM_), genomic (NC_), protein (NP_) HGVS, or rsIDs"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔬 Analyze Variant", type="primary", use_container_width=True)
    
    # Examples
    with st.expander("📋 Example Variants"):
        st.markdown("""
        - **BRCA1 missense:** `NM_007294.3:c.5266dupC`
        - **NRAS oncogenic:** `NM_002524.4:c.35G>A`
        - **CFTR common:** `NM_000492.3:c.1521_1523delCTT`
        - **dbSNP rsID:** `rs80356868`
        """)
    
    # Analysis execution
    if analyze_button:
        if not api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar")
            return
        
        if not user_query.strip():
            st.warning("Please enter a variant identifier")
            return
        
        # Initialize components
        router = GenomicQueryRouter()
        pop_filter = PopulationFilterConfig(primary_pops=primary_pops)
        
        classification = router.classify_query(user_query)
        
        if not classification.is_genomic:
            st.warning(f"❌ Could not identify genomic variant in: '{user_query}'")
            st.info("Please provide HGVS notation (e.g., NM_002496.3:c.64C>T) or rsID (e.g., rs369602258)")
            return
        
        st.success(f"✅ Detected {classification.query_type}: **{classification.extracted_identifier}**")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: ClinGen
            status_text.text("🔍 Querying ClinGen Allele Registry...")
            progress_bar.progress(10)
            
            hgvs_input = classification.extracted_identifier
            clingen_raw = query_clingen_allele(hgvs_input)
            clingen_parsed = parse_caid_minimal(clingen_raw)
            
            progress_bar.progress(25)
            
            # Step 2: VEP
            status_text.text("🧬 Retrieving VEP annotations...")
            vep_data = []
            if clingen_parsed.get('mane_ensembl'):
                vep_url = f"https://rest.ensembl.org/vep/human/hgvs/{clingen_parsed['mane_ensembl']}"
                vep_headers = {"Content-Type": "application/json"}
                vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                if vep_response.ok:
                    vep_data = vep_response.json()
            
            progress_bar.progress(50)
            
            # Step 3: MyVariant
            status_text.text("📊 Retrieving MyVariant data...")
            myv_data = {}
            if clingen_parsed.get('myvariant_hg38'):
                myv_url = f"https://myvariant.info/v1/variant/{clingen_parsed['myvariant_hg38']}?assembly=hg38"
                myv_response = requests.get(myv_url, timeout=30)
                if myv_response.ok:
                    myv_data = myv_response.json()
            
            progress_bar.progress(70)
            
            # Step 4: Parse
            status_text.text("📝 Parsing annotation data...")
            df_vep = parse_vep_list(vep_data)
            df_myvariant = parse_myv_list(myv_data)
            
            progress_bar.progress(80)
            
            # Step 5: AI Summarization
            status_text.text("🤖 Generating AI interpretation...")
            summarizer = OpenAIVariantSummarizer(api_key, model_choice, pop_filter)
            summaries = summarizer.generate_combined_summary(df_vep, df_myvariant)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.header("📋 Analysis Results")
            
            # ClinGen metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CAid", clingen_parsed.get('CAid', 'N/A'))
            with col2:
                st.metric("RSID", clingen_parsed.get('rsid', 'N/A'))
            with col3:
                st.metric("VEP Records", len(df_vep) if not df_vep.empty else 0)
            
            # VEP Summary
            st.subheader("🧬 VEP (Variant Effect Predictor) Summary")
            with st.expander("View VEP Interpretation", expanded=True):
                st.markdown(summaries['vep'])
            
            # MyVariant/ClinVar Summary
            st.subheader("🏥 MyVariant/ClinVar Summary")
            with st.expander("View Clinical Significance", expanded=True):
                st.markdown(summaries['clinvar'])
            
            # Data tables
            st.subheader("📊 Raw Data Tables")
            
            tab1, tab2 = st.tabs(["VEP Data", "MyVariant Data"])
            
            with tab1:
                if not df_vep.empty:
                    st.dataframe(df_vep, use_container_width=True)
                    st.download_button(
                        "⬇️ Download VEP CSV",
                        df_vep.to_csv(index=False),
                        f"vep_{classification.extracted_identifier}.csv",
                        "text/csv"
                    )
                else:
                    st.info("No VEP data available")
            
            with tab2:
                if not df_myvariant.empty:
                    st.dataframe(df_myvariant, use_container_width=True)
                    st.download_button(
                        "⬇️ Download MyVariant CSV",
                        df_myvariant.to_csv(index=False),
                        f"myvariant_{classification.extracted_identifier}.csv",
                        "text/csv"
                    )
                else:
                    st.info("No MyVariant data available")
            
        except requests.exceptions.RequestException as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ API Request Error: {str(e)}")
            st.info("This could be due to network issues or invalid variant notation")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis Error: {str(e)}")
            with st.expander("🐛 Debug Information"):
                st.code(str(e))

if __name__ == "__main__":
    main()
