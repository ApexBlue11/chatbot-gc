# app.py (patched) - full file
# Notes: robust MyVariant fallback for HGVS, more robust VEP POST calls, and clearer error messages.
import streamlit as st
import requests
import pandas as pd
import json
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from io import StringIO
import urllib.parse

# Configure Streamlit page
st.set_page_config(
    page_title="Genetic Variant Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (unchanged)
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
.info-box { background-color:#d1ecf1; border:1px solid #bee5eb; padding:1rem; border-radius:.5rem; }
</style>
""", unsafe_allow_html=True)

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
        q = query.strip()
        for variant_type, patterns in self.hgvs_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, q, re.IGNORECASE)
                if match:
                    return QueryClassification(True, f'hgvs_{variant_type}', match.group(0))
        rsid_match = re.search(self.rsid_pattern, q, re.IGNORECASE)
        if rsid_match:
            return QueryClassification(True, 'rsid', rsid_match.group(1))
        return QueryClassification(False, 'general', None)

def query_clingen_allele(hgvs: str) -> Dict[str, Any]:
    base_url = "https://reg.clinicalgenome.org/allele"
    params = {'hgvs': hgvs}
    with st.spinner(f"Querying ClinGen for: {hgvs}"):
        r = requests.get(base_url, params=params, timeout=30)
        # we return both status and json (or text) so UI can display diagnostics
        try:
            r.raise_for_status()
            return {'status': r.status_code, 'json': r.json()}
        except Exception as e:
            # return body text for better debugging
            body = None
            try:
                body = r.text
            except:
                body = str(e)
            return {'status': getattr(r, 'status_code', 'N/A'), 'error': str(e), 'body': body}

def parse_caid_minimal(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    # raw_json might be a dict or container returned from query_clingen_allele
    if not raw_json:
        return {}
    # If wrapped result (status/json), unwrap
    if 'json' in raw_json:
        raw = raw_json['json']
    else:
        raw = raw_json

    result: Dict[str, Any] = {}
    result['CAid'] = raw.get('@id', '').split('/')[-1] if raw.get('@id') else None
    dbsnp = raw.get('externalRecords', {}).get('dbSNP', [])
    # dbSNP entries sometimes contain 'rs' or 'rsid'; handle both
    rsid = None
    if isinstance(dbsnp, list) and dbsnp:
        first = dbsnp[0]
        rsid = first.get('rs') or first.get('rsid')
    result['rsid'] = rsid

    # genomic alleles: try to extract GRCh38/37
    result['genomic_hgvs_grch38'] = None
    result['genomic_hgvs_grch37'] = None
    for g in raw.get('genomicAlleles', []):
        ref_genome = g.get('referenceGenome', '')
        hgvs_list = g.get('hgvs') or []
        if isinstance(hgvs_list, list) and hgvs_list:
            first_h = hgvs_list[0]
        else:
            first_h = hgvs_list or None
        if 'GRCh38' in ref_genome and first_h:
            result['genomic_hgvs_grch38'] = first_h
        if 'GRCh37' in ref_genome and first_h:
            result['genomic_hgvs_grch37'] = first_h

    # External records for MyVariant (best-effort)
    ext = raw.get('externalRecords', {})
    mv_hg38 = None
    try:
        # Some records store as MyVariantInfo_hg38 or MyVariantInfo.hg38
        if 'MyVariantInfo_hg38' in ext:
            mv_hg38 = ext.get('MyVariantInfo_hg38', [{}])[0].get('id')
        elif 'MyVariantInfo.hg38' in ext:
            mv_hg38 = ext.get('MyVariantInfo.hg38', [{}])[0].get('id')
    except Exception:
        mv_hg38 = None
    result['myvariant_hg38'] = mv_hg38

    # MANE info extraction (best-effort)
    result['mane_ensembl'] = None
    transcripts = raw.get('transcriptAlleles', [])
    for t in transcripts:
        mane = t.get('MANE', {})
        if isinstance(mane, dict) and mane.get('maneStatus') == 'MANE Select':
            nuc = mane.get('nucleotide', {})
            result['mane_ensembl'] = nuc.get('Ensembl', {}).get('hgvs') or nuc.get('RefSeq', {}).get('hgvs') or None
            break

    return result

def _safe_json(r: requests.Response) -> Tuple[Optional[Any], Optional[str]]:
    """Return (json_or_none, error_text_or_none)"""
    try:
        return r.json(), None
    except Exception as e:
        return None, f"Failed to parse JSON: {str(e)}. Raw text: {r.text[:500]}"

def get_variant_annotations(clingen_data: Dict[str, Any], classification: Optional[QueryClassification] = None) -> Dict[str, Any]:
    """
    Robust annotations retrieval:
    - Try MyVariant with any available identifier (ClinGen-provided MyVariant id, RSID, HGVS string)
    - Try VEP with MANE -> direct HGVS -> Ensembl transcript fallback.
    """
    annotations: Dict[str, Any] = {'myvariant_data': None, 'vep_data': None, 'errors': [], 'debug': []}

    # Decide MyVariant query ID:
    mv_query = None
    if clingen_data:
        mv_query = clingen_data.get('myvariant_hg38') or clingen_data.get('myvariant_hg19')
        if mv_query:
            annotations['debug'].append(f"MyVariant ID from ClinGen: {mv_query}")
    # If no MyVariant ID from ClinGen, use RSID or HGVS passed in classification
    if not mv_query and classification:
        if classification.query_type == 'rsid':
            mv_query = classification.extracted_identifier  # 'rs123'
            annotations['debug'].append(f"Using RSID for MyVariant: {mv_query}")
        else:
            mv_query = classification.extracted_identifier
            annotations['debug'].append(f"Using input HGVS for MyVariant: {mv_query}")

    # Try MyVariant if we have something likely to work
    if mv_query:
        try:
            with st.spinner(f"Querying MyVariant.info with: {mv_query}"):
                # MyVariant supports variants by HGVS (e.g., "chr1:g.123A>T") or rsid.
                myv_url = f"https://myvariant.info/v1/variant/{urllib.parse.quote_plus(str(mv_query))}?assembly=hg38"
                r = requests.get(myv_url, timeout=30)
                if r.ok:
                    parsed, err = _safe_json(r)
                    if err:
                        annotations['errors'].append(f"MyVariant JSON parse error: {err}")
                    else:
                        # MyVariant sometimes wraps results in 'hits' or returns dict/list
                        if isinstance(parsed, dict) and parsed.get('hits'):
                            # If it returned a search-like structure, keep first hit
                            hits = parsed.get('hits')
                            annotations['myvariant_data'] = hits[0] if isinstance(hits, list) and hits else parsed
                        else:
                            annotations['myvariant_data'] = parsed
                        annotations['debug'].append(f"MyVariant HTTP {r.status_code}, keys: {list(annotations['myvariant_data'].keys()) if annotations['myvariant_data'] else 'None'}")
                else:
                    annotations['errors'].append(f"MyVariant query failed: HTTP {r.status_code} - {r.text[:300]}")
        except Exception as e:
            annotations['errors'].append(f"MyVariant exception: {str(e)}")

    else:
        annotations['debug'].append("No MyVariant query candidate (no ClinGen id, RSID, or HGVS).")

    # --- VEP queries (prefer POST with hgvs_notations for HGVS content) ---
    vep_result = None
    # Priority 1: if ClinGen provides a MANE or HGVS, try that as HGVS notation
    tried = []
    if clingen_data:
        mane = clingen_data.get('mane_ensembl')
        if mane:
            tried.append(('mane', mane))
    # If we have a genomic HGVS from ClinGen (explicit), try that
    if clingen_data:
        for key in ('genomic_hgvs_grch38', 'genomic_hgvs_grch37'):
            if clingen_data.get(key):
                tried.append((key, clingen_data.get(key)))
    # If classification contains the original HGVS, try it
    if classification and classification.extracted_identifier:
        tried.append(('input', classification.extracted_identifier))

    # Also, if MyVariant returned a useful vep-like field or vcf, try to use it
    mv = annotations.get('myvariant_data')
    if mv and isinstance(mv, dict):
        # myvariant often contains 'vcf' or 'hgvs' fields; record them for debugging
        if mv.get('vcf'):
            tried.append(('myvariant.vcf', mv.get('vcf')))
        if mv.get('hgvs'):
            tried.append(('myvariant.hgvs', mv.get('hgvs')))

    # Deduplicate tried entries preserving order
    seen_vals = set()
    tried_unique = []
    for tag, val in tried:
        sval = str(val)
        if sval not in seen_vals:
            tried_unique.append((tag, val))
            seen_vals.add(sval)

    # Attempt VEP via POST hgvs_notations for each candidate that *looks like* HGVS or a transcript: prefer strings only
    vep_errors = []
    for tag, candidate in tried_unique:
        # if candidate is dict/list and contains usable strings, skip here (we handle separately)
        if not candidate:
            continue
        if isinstance(candidate, (dict, list)):
            # try to extract a string HGVS inside
            if isinstance(candidate, dict):
                # look for common keys
                for k in ['hgvs', 'hgvsc', 'hgvsp', 'coding']:
                    if k in candidate:
                        candidate = candidate[k]
                        break
            elif isinstance(candidate, list) and candidate:
                candidate = candidate[0]
            # still might be non-string; convert
            candidate = str(candidate)

        # Sanity: skip if too long or obviously not an HGVS-like string and not rsid
        if not isinstance(candidate, str):
            continue

        # Decide whether to use VEP hgvs endpoint (if contains ":" or "g." etc) or to call id endpoint
        as_hgvs = False
        if ':' in candidate or 'g.' in candidate or candidate.lower().startswith('chr'):
            as_hgvs = True

        if as_hgvs:
            # POST - robust
            try:
                with st.spinner(f"Querying Ensembl VEP (hgvs) for [{tag}]"):
                    vep_url = "https://rest.ensembl.org/vep/human/hgvs"
                    body = {"hgvs_notations": [candidate]}
                    headers = {"Content-Type": "application/json", "Accept": "application/json"}
                    r = requests.post(vep_url, headers=headers, json=body, timeout=60)
                    if r.ok:
                        parsed, err = _safe_json(r)
                        if parsed:
                            vep_result = parsed
                            annotations['debug'].append(f"VEP succeeded for [{tag}] via hgvs (HTTP {r.status_code})")
                            break
                        else:
                            vep_errors.append(f"VEP parse error for {tag}: {err}")
                    else:
                        vep_errors.append(f"VEP HTTP {r.status_code} for {tag}: {r.text[:300]}")
            except Exception as e:
                vep_errors.append(f"VEP exception for {tag}: {str(e)}")
        else:
            # Try VEP by id (IDs: transcript IDs or rsIDs)
            try:
                with st.spinner(f"Querying Ensembl VEP (id) for [{tag}]"):
                    # Use the generic VEP ID endpoint
                    vep_url = f"https://rest.ensembl.org/vep/human/id/{urllib.parse.quote_plus(candidate)}"
                    headers = {"Content-Type": "application/json", "Accept": "application/json"}
                    r = requests.get(vep_url, headers=headers, timeout=30)
                    if r.ok:
                        parsed, err = _safe_json(r)
                        if parsed:
                            vep_result = parsed
                            annotations['debug'].append(f"VEP succeeded for [{tag}] via id (HTTP {r.status_code})")
                            break
                        else:
                            vep_errors.append(f"VEP parse error for id {tag}: {err}")
                    else:
                        vep_errors.append(f"VEP HTTP {r.status_code} for id {tag}: {r.text[:300]}")
            except Exception as e:
                vep_errors.append(f"VEP exception for id {tag}: {str(e)}")

    # If vep_result still None but MyVariant contains a vep-like field, take it
    if not vep_result and mv:
        # some myvariant responses include a 'vep' or 'vcf' or 'vep' keys
        if mv.get('vcf'):
            annotations['vep_data'] = mv.get('vcf')
            annotations['debug'].append("Using myvariant.vcf as vep-like data")
        elif mv.get('vep'):
            annotations['vep_data'] = mv.get('vep')
            annotations['debug'].append("Using myvariant.vep as vep-like data")
    else:
        annotations['vep_data'] = vep_result

    # Consolidate vep_errors and push into errors for visibility
    if vep_errors:
        for e in vep_errors:
            annotations['errors'].append(e)

    return annotations

# --- Display helpers (kept similar to yours but referencing the patched annotations structure) ---

def select_primary_vep_transcript(vep_data):
    if not vep_data or not isinstance(vep_data, list) or not vep_data[0].get('transcript_consequences'):
        return None, "No transcript consequences found"
    transcripts = vep_data[0]['transcript_consequences']
    for t in transcripts:
        flags = t.get('flags', [])
        if isinstance(flags, list) and ('MANE_SELECT' in flags or any('mane' in str(f).lower() for f in flags)):
            return t, "MANE Select"
    for t in transcripts:
        if t.get('canonical') == 1 or ('canonical' in t.get('flags', [])):
            return t, "Canonical"
    for t in transcripts:
        if t.get('biotype') == 'protein_coding' and 'missense_variant' in t.get('consequence_terms', []):
            return t, "Protein coding with missense"
    for t in transcripts:
        if t.get('biotype') == 'protein_coding':
            return t, "First protein coding"
    return transcripts[0], "First available transcript"

def display_vep_analysis(vep_data):
    if not vep_data:
        st.warning("No VEP data available")
        return
    try:
        primary, reason = select_primary_vep_transcript(vep_data)
    except Exception as e:
        st.error("Error selecting primary transcript: " + str(e))
        st.json(vep_data)
        return
    if primary:
        st.subheader("Primary transcript")
        st.write(f"Selection reason: {reason}")
        st.write(f"Transcript ID: {primary.get('transcript_id', 'N/A')}")
        st.write(f"Gene: {primary.get('gene_symbol', 'N/A')}")
        st.write("Consequence terms: " + ", ".join(primary.get('consequence_terms', [])))
    with st.expander("Full VEP JSON (expand)"):
        st.json(vep_data)

def display_comprehensive_myvariant_data(myvariant_data):
    if not myvariant_data:
        st.warning("No MyVariant data available")
        return
    st.subheader("MyVariant.info summary")
    if isinstance(myvariant_data, dict):
        # present some key fields
        st.write("Gene:", myvariant_data.get('genename') or myvariant_data.get('gene') or myvariant_data.get('symbol'))
        st.write("rsid:", myvariant_data.get('rsid') or myvariant_data.get('dbsnp', {}).get('rsid'))
    with st.expander("Full MyVariant JSON"):
        st.json(myvariant_data)

# Population frequency collector (unchanged)
DEFAULT_POPULATION_LABELS = [
    'Overall', 'African', 'Latino', 'East Asian', 'South Asian', 'Non-Finnish European',
    'Finnish', 'Ashkenazi Jewish', 'Middle Eastern', 'Other', 'Amish'
]

def collect_population_frequencies(myvariant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    if not myvariant_data or not isinstance(myvariant_data, dict):
        return records
    def add_record(src, pop, freq, ac=None, an=None, path=None):
        try:
            fv = float(freq) if freq is not None else None
        except:
            fv = None
        if fv is not None:
            records.append({'Source': src, 'Population': pop, 'Frequency': fv, 'AC': ac or 'N/A', 'AN': an or 'N/A', 'Path': path or ''})
    ge = myvariant_data.get('gnomad_exome', {}) or {}
    if isinstance(ge, dict):
        af = ge.get('af', {}) if isinstance(ge.get('af', {}), dict) else ge.get('af')
        ac = ge.get('ac', {}) or {}
        an = ge.get('an', {}) or {}
        pops = {'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino', 'af_asj': 'Ashkenazi Jewish','af_eas': 'East Asian','af_fin': 'Finnish','af_nfe': 'Non-Finnish European','af_sas': 'South Asian','af_oth': 'Other'}
        if isinstance(af, dict):
            for k, name in pops.items():
                val = af.get(k)
                add_record('gnomAD Exome', name, val, ac.get(k.replace('af','ac')) if isinstance(ac, dict) else None, an.get(k.replace('af','an')) if isinstance(an, dict) else None, f'gnomad_exome.af.{k}')
        else:
            add_record('gnomAD Exome', 'Overall', af, ac if not isinstance(ac, dict) else ac.get('ac'), an if not isinstance(an, dict) else an.get('an'), 'gnomad_exome.af')
    # gnomad_genome, 1000G, ExAC are similar — reuse code in your original app or add as needed
    return records

def create_download_section(clingen_data, myvariant_data, vep_data, classification):
    st.subheader("📥 Download Data")
    c1, c2, c3 = st.columns(3)
    if c1:
        if clingen_data:
            try:
                st.download_button("ClinGen JSON", json.dumps(clingen_data, indent=2), file_name="clingen.json", mime="application/json")
            except Exception as e:
                st.write("Download ClinGen failed: " + str(e))
    if c2:
        if myvariant_data:
            try:
                st.download_button("MyVariant JSON", json.dumps(myvariant_data, indent=2), file_name="myvariant.json", mime="application/json")
            except Exception as e:
                st.write("Download MyVariant failed: " + str(e))
    if c3:
        if vep_data:
            try:
                st.download_button("VEP JSON", json.dumps(vep_data, indent=2), file_name="vep.json", mime="application/json")
            except Exception as e:
                st.write("Download VEP failed: " + str(e))

# --- Main UI ---
def main():
    st.markdown('<h1 class="main-header">🧬 Genetic Variant Analyzer (patched)</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Examples")
        if st.button("Load example: NDUFS8 (HGVS)"):
            st.session_state['example_input'] = "NM_002496.3:c.64C>T"
        if st.button("Load example: BRCA1 (HGVS)"):
            st.session_state['example_input'] = "NM_007294.3:c.5266dupC"
        if st.button("Load example: rsID"):
            st.session_state['example_input'] = "rs121913529"

    default_value = st.session_state.pop('example_input', "") if 'example_input' in st.session_state else ""
    user_input = st.text_input("Enter HGVS or RSID:", value=default_value, placeholder="NM_002496.3:c.64C>T or rs123...", key="variant_input")
    analyze = st.button("🔬 Analyze Variant")
    if analyze and not user_input:
        st.error("Please provide input to analyze")
        st.stop()

    # Run analysis if requested or if existing session data exists for same query
    should_analyze = analyze and user_input
    use_existing = False
    if 'analysis_data' in st.session_state and st.session_state.get('last_query') == user_input and not should_analyze:
        use_existing = True

    if should_analyze:
        router = GenomicQueryRouter()
        classification = router.classify_query(user_input)
        if not classification.is_genomic:
            st.error("Invalid query. Provide an HGVS notation (e.g. NM_...:c...) or an RSID (rs...).")
            st.stop()

        # Try ClinGen for HGVS queries; skip for rsid
        clingen_data = {}
        annotations = {'myvariant_data': None, 'vep_data': None, 'errors': [], 'debug': []}
        try:
            if classification.query_type == 'rsid':
                # RSID flow
                clingen_data = {'CAid': None, 'rsid': classification.extracted_identifier}
                annotations = get_variant_annotations(clingen_data, classification)
            else:
                # HGVS flow: query ClinGen first, but don't trust it to always give myvariant id
                cg_resp = query_clingen_allele(classification.extracted_identifier)
                if cg_resp is None:
                    st.warning("ClinGen returned no response")
                    clingen_data = {}
                elif 'error' in cg_resp:
                    # show ClinGen error but proceed to try MyVariant directly using input HGVS
                    st.warning(f"ClinGen lookup error (status {cg_resp.get('status')}): {cg_resp.get('error')}")
                    # attach raw body for debugging
                    st.info(f"ClinGen raw body (first 500 chars): {cg_resp.get('body')[:500] if cg_resp.get('body') else 'N/A'}")
                    clingen_data = {}
                    # still attempt to query MyVariant/VEP using the HGVS input
                    annotations = get_variant_annotations({}, classification)
                else:
                    # parse and proceed
                    clingen_data = parse_caid_minimal(cg_resp)
                    annotations = get_variant_annotations(clingen_data, classification)
        except Exception as e:
            st.error("Analysis failed: " + str(e))
            st.stop()

        # save to session state
        st.session_state['analysis_data'] = {
            'classification': classification,
            'clingen_data': clingen_data,
            'annotations': annotations,
            'query_time': time.time()
        }
        st.session_state['last_query'] = user_input
        use_existing = True

    if use_existing:
        analysis = st.session_state['analysis_data']
        classification = analysis['classification']
        clingen_data = analysis['clingen_data']
        annotations = analysis['annotations']

        # Show debug and error information up front for transparency
        if annotations.get('debug'):
            with st.expander("Debug trace (what the app tried)"):
                for d in annotations.get('debug', []):
                    st.write("- " + str(d))
        if annotations.get('errors'):
            with st.expander("API errors / warnings (click to expand)"):
                for err in annotations.get('errors', []):
                    st.warning(err)

        # Basic identifiers
        st.markdown('<div class="section-header">Identifiers</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Query:**", classification.extracted_identifier)
            st.write("**Type:**", classification.query_type)
        with col2:
            st.write("**ClinGen CAid:**", clingen_data.get('CAid', "N/A"))
            st.write("**ClinGen RSID:**", clingen_data.get('rsid', "N/A"))

        # Main tabs
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.tabs(["VEP", "MyVariant", "Population / Clinical", "Raw responses"])

        with t1:
            if annotations.get('vep_data'):
                display_vep_analysis(annotations['vep_data'])
            else:
                st.info("No VEP data available. Check errors tab for why.

If VEP failed, the app attempted multiple fallbacks. Expand the 'API errors / warnings' box above for details.")

        with t2:
            if annotations.get('myvariant_data'):
                display_comprehensive_myvariant_data(annotations['myvariant_data'])
            else:
                st.info("No MyVariant results. Either MyVariant didn't have a record for this variant or the query failed. See errors above.")

        with t3:
            st.subheader("Population Frequency Summary")
            mv = annotations.get('myvariant_data') or {}
            records = collect_population_frequencies(mv)
            if records:
                df = pd.DataFrame(records).sort_values('Frequency', ascending=False).reset_index(drop=True)
                st.dataframe(df, use_container_width=True)
                try:
                    st.bar_chart(df.set_index('Population')['Frequency'])
                except:
                    pass
            else:
                st.info("No population frequency records found in MyVariant.")

            st.markdown("---")
            st.subheader("Clinical / ClinVar summary (if available)")
            clin = (mv.get('clinvar') if isinstance(mv, dict) else None) or {}
            if clin:
                st.write("Clinical significance (ClinVar):", clin.get('clinical_significance') or clin.get('clnsig') or "N/A")
                rcv = clin.get('rcv') or []
                if rcv:
                    st.write(f"{len(rcv)} ClinVar submission records. Expand Raw responses -> MyVariant for full details.")
            else:
                st.info("No ClinVar info in MyVariant response.")

        with t4:
            st.subheader("Raw API responses captured")
            if clingen_data:
                with st.expander("ClinGen (parsed minimal)"):
                    st.json(clingen_data)
            if annotations.get('myvariant_data'):
                with st.expander("MyVariant.info raw"):
                    st.json(annotations.get('myvariant_data'))
            if annotations.get('vep_data'):
                with st.expander("VEP raw"):
                    st.json(annotations.get('vep_data'))

            create_download_section(clingen_data, annotations.get('myvariant_data'), annotations.get('vep_data'), classification)

        st.success("Analysis complete — check the 'API errors / warnings' expander for anything suspicious.")

    # validation-only display (no analyze pressed)
    elif user_input and not analyze:
        router = GenomicQueryRouter()
        classification = router.classify_query(user_input)
        if classification.is_genomic:
            st.info(f"Valid input detected: {classification.query_type} ({classification.extracted_identifier})")
        else:
            st.error("Invalid format. Please enter HGVS or rsID.")

    # Footer note
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666;">
    <small>Patched runner: MyVariant + Ensembl VEP fallbacks added, plus explicit debug/error traces to help diagnose missing data.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
