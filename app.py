# Add the original display functions here - these are COMPLETE and UNTRUNCATED
def select_primary_vep_transcript(vep_data):
    """Select the primary transcript for VEP analysis based on priority."""
    if not vep_data or not vep_data[0].get('transcript_consequences'):
        return None, "No transcript consequences found"
    
    transcripts = vep_data[0]['transcript_consequences']
    
    # Priority 1: MANE Select transcript
    for t in transcripts:
        flags = t.get('flags', [])
        if 'MANE_SELECT' in flags or any('mane' in str(flag).lower() for flag in flags):
            return t, "MANE Select"
    
    # Priority 2: Canonical transcript
    for t in transcripts:
        if t.get('canonical') == 1 or 'canonical' in t.get('flags', []):
            return t, "Canonical"
    
    # Priority 3: First protein coding transcript with complete annotations
    for t in transcripts:
        if (t.get('biotype') == 'protein_coding' and 
            'missense_variant' in t.get('consequence_terms', [])):
            return t, "First protein coding with missense annotation"
    
    # Priority 4: Any protein coding transcript
    for t in transcripts:
        if t.get('biotype') == 'protein_coding':
            return t, "First protein coding"
    
    return transcripts[0], "First available transcript"

def display_vep_analysis_original(vep_data):
    """Display comprehensive VEP analysis - ORIGINAL FUNCTION."""
    if not vep_data or not vep_data[0].get('transcript_consequences'):
        st.warning("No VEP data available")
        return
    
    variant_info = vep_data[0]
    all_transcripts = variant_info.get('transcript_consequences', [])
    
    # Select primary transcript
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
                if transcript.get('canonical') == 1:
                    special_flags.append("CANONICAL")
                if 'MANE_SELECT' in flags:
                    special_flags.append("MANE SELECT")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Gene:** {transcript.get('gene_symbol', 'N/A')}")
                    if special_flags:
                        st.success(f"🏷️ {', '.join(special_flags)}")
                with col2:
                    st.write(f"**Consequence:** {', '.join(transcript.get('consequence_terms', []))}")
                    st.write(f"**Impact:** {transcript.get('impact', 'N/A')}")
                with col3:
                    st.write(f"**Biotype:** {transcript.get('biotype', 'N/A')}")
                    if transcript.get('distance'):
                        st.write(f"**Distance:** {transcript.get('distance', 'N/A')}")
                with col4:
                    if transcript.get('amino_acids'):
                        st.write(f"**AA Change:** {transcript.get('amino_acids', 'N/A')}")
                        st.write(f"**Position:** {transcript.get('protein_start', 'N/A')}")
                
                if transcript.get('sift_score') or transcript.get('polyphen_score'):
                    pred_col1, pred_col2 = st.columns(2)
                    with pred_col1:
                        if transcript.get('sift_score'):
                            st.write(f"**SIFT:** {transcript['sift_score']:.3f} ({transcript.get('sift_prediction', 'N/A')})")
                    with pred_col2:
                        if transcript.get('polyphen_score'):
                            st.write(f"**PolyPhen:** {transcript['polyphen_score']:.3f} ({transcript.get('polyphen_prediction', 'N/A')})")
                
                st.markdown("---")

def display_comprehensive_myvariant_data_original(myvariant_data):
    """Display comprehensive MyVariant.info data analysis - ORIGINAL FUNCTION."""
    if not myvariant_data:
        st.warning("No MyVariant data available")
        return
    
    if isinstance(myvariant_data, list):
        if len(myvariant_data) > 1:
            st.info(f"Multiple variants found ({len(myvariant_data)}). Showing first result.")
        if len(myvariant_data) > 0:
            myvariant_data = myvariant_data[0]
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
        
        chrom = (myvariant_data.get('hg38', {}).get('chr') or 
                myvariant_data.get('chrom') or 'N/A')
        
        hg38_data = myvariant_data.get('hg38', {})
        pos = (hg38_data.get('start') or hg38_data.get('end') or hg38_data.get('pos') or
              myvariant_data.get('pos') or myvariant_data.get('vcf', {}).get('position') or 'N/A')
        
        ref = (myvariant_data.get('hg38', {}).get('ref') or 
              myvariant_data.get('ref') or 
              myvariant_data.get('vcf', {}).get('ref') or 'N/A')
        alt = (myvariant_data.get('hg38', {}).get('alt') or 
              myvariant_data.get('alt') or 
              myvariant_data.get('vcv', {}).get('alt') or 'N/A')
        
        with col1:
            st.write(f"**Chromosome:** {chrom}")
            st.write(f"**Position (hg38):** {pos}")
        with col2:
            st.write(f"**Reference:** {ref}")
            st.write(f"**Alternate:** {alt}")
        with col3:
            gene_name = (myvariant_data.get('genename') or 
                       myvariant_data.get('gene') or 
                       myvariant_data.get('symbol') or 'N/A')
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
            "Pathogenicity Predictors": [
                ("SIFT", ["sift", "score"], ["sift", "pred"]),
                ("PolyPhen2 HDiv", ["polyphen2", "hdiv", "score"], ["polyphen2", "hdiv", "pred"]),
                ("PolyPhen2 HVar", ["polyphen2", "hvar", "score"], ["polyphen2", "hvar", "pred"]),
                ("FATHMM", ["fathmm", "score"], ["fathmm", "pred"]),
                ("MutationTaster", ["mutationtaster", "score"], ["mutationtaster", "pred"]),
                ("MutationAssessor", ["mutationassessor", "score"], ["mutationassessor", "pred"]),
                ("PROVEAN", ["provean", "score"], ["provean", "pred"]),
                ("MetaSVM", ["metasvm", "score"], ["metasvm", "pred"]),
                ("MetaLR", ["metalr", "score"], ["metalr", "pred"]),
                ("M-CAP", ["m-cap", "score"], ["m-cap", "pred"]),
                ("REVEL", ["revel", "score"], None),
                ("MutPred", ["mutpred", "score"], None),
                ("LRT", ["lrt", "score"], ["lrt", "pred"]),
            ],
            "Conservation Scores": [
                ("GERP++ NR", ["gerp++", "nr"], None),
                ("GERP++ RS", ["gerp++", "rs"], None),
                ("PhyloP 100way Vertebrate", ["phylop", "100way_vertebrate", "score"], None),
                ("PhyloP 470way Mammalian", ["phylop", "470way_mammalian", "score"], None),
                ("PhastCons 100way Vertebrate", ["phastcons", "100way_vertebrate", "score"], None),
                ("PhastCons 470way Mammalian", ["phastcons", "470way_mammalian", "score"], None),
                ("SiPhy 29way", ["siphy_29way", "logodds_score"], None),
            ],
            "Ensemble Predictors": [
                ("CADD Phred", ["cadd", "phred"], None),
                ("DANN", ["dann", "score"], None),
                ("Eigen PC Phred", ["eigen-pc", "phred_coding"], None),
                ("FATHMM-MKL", ["fathmm-mkl", "coding_score"], ["fathmm-mkl", "coding_pred"]),
                ("FATHMM-XF", ["fathmm-xf", "coding_score"], ["fathmm-xf", "coding_pred"]),
                ("GenoCanyon", ["genocanyon", "score"], None),
                ("Integrated FitCons", ["fitcons", "integrated", "score"], None),
                ("VEST4", ["vest4", "score"], None),
                ("MVP", ["mvp", "score"], None),
            ],
            "Deep Learning": [
                ("PrimateAI", ["primateai", "score"], ["primateai", "pred"]),
                ("DEOGEN2", ["deogen2", "score"], ["deogen2", "pred"]),
                ("BayesDel AddAF", ["bayesdel", "add_af", "score"], ["bayesdel", "add_af", "pred"]),
                ("ClinPred", ["clinpred", "score"], ["clinpred", "pred"]),
                ("LIST-S2", ["list-s2", "score"], ["list-s2", "pred"]),
                ("AlphaMissense", ["alphamissense", "score"], ["alphamissense", "pred"]),
                ("ESM1b", ["esm1b", "score"], ["esm1b", "pred"]),
            ]
        }
        
        for category, predictors in prediction_categories.items():
            st.markdown(f"#### {category}")
            
            predictor_data = []
            
            for predictor_info in predictors:
                if len(predictor_info) == 3:
                    predictor_name, score_path, pred_path = predictor_info
                else:
                    continue
                
                score_val = extract_nested_value(dbnsfp, score_path)
                if isinstance(score_val, list) and score_val:
                    score_val = score_val[0]
                
                pred_val = None
                if pred_path:
                    pred_val = extract_nested_value(dbnsfp, pred_path)
                    if isinstance(pred_val, list) and pred_val:
                        pred_val = pred_val[0]
                
                if score_val is not None:
                    predictor_data.append({
                        'Predictor': predictor_name,
                        'Score': score_val,
                        'Prediction': pred_val or 'N/A'
                    })
            
            if predictor_data:
                cols = st.columns(3)
                for i, pred in enumerate(predictor_data):
                    col_idx = i % 3
                    with cols[col_idx]:
                        if isinstance(pred['Score'], (int, float)):
                            score_str = f"{pred['Score']:.3f}" if isinstance(pred['Score'], float) else str(pred['Score'])
                        else:
                            score_str = str(pred['Score'])
                        
                        prediction_text = pred['Prediction']
                        if prediction_text in ['D', 'Damaging', 'DAMAGING']:
                            prediction_color = "🔴"
                        elif prediction_text in ['T', 'Tolerated', 'TOLERATED', 'B', 'Benign']:
                            prediction_color = "🟢" 
                        elif prediction_text in ['P', 'Possibly damaging', 'POSSIBLY_DAMAGING']:
                            prediction_color = "🟡"
                        else:
                            prediction_color = ""
                        
                        display_pred = f"{prediction_color} {prediction_text}" if prediction_color else prediction_text
                        
                        st.metric(
                            pred['Predictor'],
                            score_str,
                            delta=display_pred if display_pred != 'N/A' else None
                        )
            else:
                st.info(f"No {category.lower()} data available")
    
    with data_tabs[2]:  # Population Frequencies
        st.subheader("Population Frequency Data")
        
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
                                'Total Alleles': an or 'N/A'
                            })
                    
                    if pop_data:
                        df_freq = pd.DataFrame(pop_data)
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
                        'af': 'Overall', 'af_afr': 'African', 'af_amr': 'Latino',
                        'af_ami': 'Amish', 'af_asj': 'Ashkenazi Jewish', 
                        'af_eas': 'East Asian', 'af_fin': 'Finnish', 
                        'af_mid': 'Middle Eastern', 'af_nfe': 'Non-Finnish European',
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
                                'Total Alleles': an or 'N/A'
                            })
                    
                    if pop_data:
                        df_freq = pd.DataFrame(pop_data)
                        st.dataframe(df_freq, use_container_width=True)
                        
                        chart_data = df_freq[df_freq['Frequency'] > 0].set_index('Population')['Frequency']
                        if not chart_data.empty:
                            st.bar_chart(chart_data)
                    else:
                        st.info("No gnomAD genome frequency data above threshold")
        
        with freq_tabs[2]:  # 1000 Genomes
            kg_data = myvariant_data.get('dbnsfp', {}).get('1000gp3', {})
            if kg_data:
                st.markdown("**1000 Genomes Project Phase 3**")
                
                overall_freq = kg_data.get('af')
                overall_ac = kg_data.get('ac')
                
                if overall_freq and overall_freq > 0:
                    st.write(f"**Overall Frequency:** {overall_freq:.6f}")
                    st.write(f"**Overall Allele Count:** {overall_ac}")
                    
                    pop_data = []
                    populations = {
                        'afr': 'African', 'amr': 'American', 'eas': 'East Asian',
                        'eur': 'European', 'sas': 'South Asian'
                    }
                    
                    for pop_key, pop_name in populations.items():
                        pop_info = kg_data.get(pop_key, {})
                        if isinstance(pop_info, dict):
                            freq = pop_info.get('af')
                            ac = pop_info.get('ac')
                            if freq and freq > 0:
                                pop_data.append({
                                    'Population': pop_name,
                                    'Frequency': freq,
                                    'Allele Count': ac
                                })
                    
                    if pop_data:
                        df_pop = pd.DataFrame(pop_data)
                        st.dataframe(df_pop, use_container_width=True)
            else:
                st.info("No 1000 Genomes data available")
        
        with freq_tabs[3]:  # ExAC
            exac_data = myvariant_data.get('dbnsfp', {}).get('exac', {})
            if exac_data:
                st.markdown("**Exome Aggregation Consortium (ExAC)**")
                
                overall_freq = exac_data.get('af')
                overall_ac = exac_data.get('ac')
                
                if overall_freq and overall_freq > 0:
                    st.write(f"**Overall Frequency:** {overall_freq:.6f}")
                    st.write(f"**Overall AC:** {overall_ac}")
                    
                    populations = {
                        'afr': 'African', 'amr': 'Latino', 'eas': 'East Asian',
                        'fin': 'Finnish', 'nfe': 'Non-Finnish European', 'sas': 'South Asian'
                    }
                    
                    pop_data = []
                    for pop_key, pop_name in populations.items():
                        pop_freq = exac_data.get(pop_key)
                        if isinstance(pop_freq, dict):
                            freq = pop_freq.get('af')
                        else:
                            freq = pop_freq
                        
                        if freq and freq > 0:
                            pop_data.append({
                                'Population': pop_name,
                                'Frequency': freq
                            })
                    
                    if pop_data:
                        df_pop = pd.DataFrame(pop_data)
                        st.dataframe(df_pop, use_container_width=True)
        
        with freq_tabs[4]:  # Raw frequency data
            st.markdown("**All Available Frequency Fields**")
            
            freq_fields = {}
            
            def collect_freq_fields(data, prefix=""):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if 'af' in key.lower() or 'freq' in key.lower():
                        if isinstance(value, (int, float)) and value > 0:
                            freq_fields[full_key] = value
                    elif isinstance(value, dict):
                        collect_freq_fields(value, full_key)
            
            collect_freq_fields(myvariant_data)
            
            if freq_fields:
                freq_df = pd.DataFrame([
                    {'Field': k, 'Frequency': v} for k, v in sorted(freq_fields.items())
                ])
                st.dataframe(freq_df, use_container_width=True)
            else:
                st.info("No frequency fields found")
    
    with data_tabs[3]:  # ClinVar
        st.subheader("ClinVar Clinical Annotations")
        
        clinvar_data = myvariant_data.get('clinvar', {})
        if not clinvar_data:
            st.info("No ClinVar data available")
            return
        
        clinical_sig = (clinvar_data.get('clinical_significance') or 
                       clinvar_data.get('clnsig') or 'N/A')
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Clinical Significance:** {clinical_sig}")
            
            if clinvar_data.get('variant_id'):
                st.write(f"**Variation ID:** {clinvar_data['variant_id']}")
            if clinvar_data.get('allele_id'):
                st.write(f"**Allele ID:** {clinvar_data['allele_id']}")
        
        with col2:
            gene_info = clinvar_data.get('gene', {})
            if isinstance(gene_info, dict):
                if gene_info.get('symbol'):
                    st.write(f"**Gene Symbol:** {gene_info['symbol']}")
                if gene_info.get('id'):
                    st.write(f"**Gene ID:** {gene_info['id']}")
        
        hgvs_info = clinvar_data.get('hgvs', {})
        if isinstance(hgvs_info, dict):
            st.subheader("HGVS Notations")
            col1, col2 = st.columns(2)
            with col1:
                if hgvs_info.get('coding'):
                    st.write(f"**Coding:** {hgvs_info['coding']}")
                if hgvs_info.get('protein'):
                    st.write(f"**Protein:** {hgvs_info['protein']}")
            with col2:
                if hgvs_info.get('genomic'):
                    genomic = hgvs_info['genomic']
                    if isinstance(genomic, list):
                        st.write(f"**Genomic:** {', '.join(genomic)}")
                    else:
                        st.write(f"**Genomic:** {str(genomic)}")
        
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
                                    id_list = []
                                    for db, id_val in identifiers.items():
                                        id_list.append(f"{db}: {id_val}")
                                    st.write(f"**Identifiers:** {', '.join(id_list)}")
    
    with data_tabs[4]:  # External DBs
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
            if uniprot_data.get('clinical_significance'):
                st.write(f"**Clinical Significance:** {uniprot_data['clinical_significance']}")
            if uniprot_data.get('source_db_id'):
                st.write(f"**Source DB ID:** {uniprot_data['source_db_id']}")

if __name__ == "__main__":
    main()import streamlit as st
import requests
import pandas as pd
import json
import time
import re
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from io import StringIO
import openai

# Configure Streamlit page
st.set_page_config(
    page_title="Genetic Variant Analyzer with AI Assistant",
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
.ai-response-box {
    background-color: #f0f7ff;
    border: 2px solid #4a90e2;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 1rem 0;
}
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
    """Query ClinGen Allele Registry by HGVS notation."""
    base_url = "http://reg.clinicalgenome.org/allele"
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

def get_variant_annotations(clingen_data, classification=None):
    """Retrieve variant annotations from multiple APIs."""
    annotations = {
        'myvariant_data': {},
        'vep_data': [],
        'errors': []
    }
    
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
    
    if (classification and classification.query_type == 'rsid' and 
        not annotations['vep_data'] and not vep_attempted):
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
    
    return annotations

def extract_population_frequencies(myvariant_data, selected_populations):
    """Extract and filter population frequency data based on user selection."""
    freq_data = {
        'gnomad_exome': {},
        'gnomad_genome': {},
        '1000genomes': {},
        'exac': {}
    }
    
    if not myvariant_data:
        return freq_data
    
    # Handle list response
    if isinstance(myvariant_data, list) and len(myvariant_data) > 0:
        myvariant_data = myvariant_data[0]
    
    if not isinstance(myvariant_data, dict):
        return freq_data
    
    # Population mapping
    pop_mapping = {
        'African/African American': ['afr', 'af_afr'],
        'Latino/Admixed American': ['amr', 'af_amr'],
        'Ashkenazi Jewish': ['asj', 'af_asj'],
        'East Asian': ['eas', 'af_eas'],
        'Finnish': ['fin', 'af_fin'],
        'Non-Finnish European': ['nfe', 'af_nfe'],
        'South Asian': ['sas', 'af_sas'],
        'Middle Eastern': ['mid', 'af_mid'],
        'Amish': ['ami', 'af_ami'],
        'Other': ['oth', 'af_oth'],
        'Overall': ['af', 'af']
    }
    
    # Extract gnomAD exome frequencies
    gnomad_exome = myvariant_data.get('gnomad_exome', {})
    if gnomad_exome:
        af_data = gnomad_exome.get('af', {})
        an_data = gnomad_exome.get('an', {})
        ac_data = gnomad_exome.get('ac', {})
        
        for pop_name, pop_keys in pop_mapping.items():
            if pop_name in selected_populations or 'All Populations' in selected_populations:
                for key in pop_keys:
                    if key in af_data and af_data[key]:
                        freq_data['gnomad_exome'][pop_name] = {
                            'frequency': af_data[key],
                            'allele_count': ac_data.get(key.replace('af', 'ac'), 'N/A'),
                            'allele_number': an_data.get(key.replace('af', 'an'), 'N/A')
                        }
                        break
    
    # Extract gnomAD genome frequencies
    gnomad_genome = myvariant_data.get('gnomad_genome', {})
    if gnomad_genome:
        af_data = gnomad_genome.get('af', {})
        an_data = gnomad_genome.get('an', {})
        ac_data = gnomad_genome.get('ac', {})
        
        for pop_name, pop_keys in pop_mapping.items():
            if pop_name in selected_populations or 'All Populations' in selected_populations:
                for key in pop_keys:
                    if key in af_data and af_data[key]:
                        freq_data['gnomad_genome'][pop_name] = {
                            'frequency': af_data[key],
                            'allele_count': ac_data.get(key.replace('af', 'ac'), 'N/A'),
                            'allele_number': an_data.get(key.replace('af', 'an'), 'N/A')
                        }
                        break
    
    # Extract 1000 Genomes frequencies
    kg_data = myvariant_data.get('dbnsfp', {}).get('1000gp3', {})
    if kg_data:
        kg_pop_mapping = {
            'African/African American': 'afr',
            'Latino/Admixed American': 'amr',
            'East Asian': 'eas',
            'Non-Finnish European': 'eur',
            'South Asian': 'sas',
            'Overall': None
        }
        
        for pop_name, pop_key in kg_pop_mapping.items():
            if pop_name in selected_populations or 'All Populations' in selected_populations:
                if pop_key is None:
                    # Overall frequency
                    if kg_data.get('af'):
                        freq_data['1000genomes'][pop_name] = {
                            'frequency': kg_data['af'],
                            'allele_count': kg_data.get('ac', 'N/A')
                        }
                else:
                    pop_info = kg_data.get(pop_key, {})
                    if isinstance(pop_info, dict) and pop_info.get('af'):
                        freq_data['1000genomes'][pop_name] = {
                            'frequency': pop_info['af'],
                            'allele_count': pop_info.get('ac', 'N/A')
                        }
    
    # Extract ExAC frequencies
    exac_data = myvariant_data.get('dbnsfp', {}).get('exac', {})
    if exac_data:
        exac_pop_mapping = {
            'African/African American': 'afr',
            'Latino/Admixed American': 'amr',
            'East Asian': 'eas',
            'Finnish': 'fin',
            'Non-Finnish European': 'nfe',
            'South Asian': 'sas',
            'Overall': None
        }
        
        for pop_name, pop_key in exac_pop_mapping.items():
            if pop_name in selected_populations or 'All Populations' in selected_populations:
                if pop_key is None:
                    if exac_data.get('af'):
                        freq_data['exac'][pop_name] = {
                            'frequency': exac_data['af'],
                            'allele_count': exac_data.get('ac', 'N/A')
                        }
                else:
                    pop_freq = exac_data.get(pop_key)
                    if isinstance(pop_freq, dict):
                        freq = pop_freq.get('af')
                    else:
                        freq = pop_freq
                    
                    if freq:
                        freq_data['exac'][pop_name] = {
                            'frequency': freq,
                            'allele_count': 'N/A'
                        }
    
    return freq_data

def prepare_ai_context(clingen_data, vep_data, myvariant_data, population_freq_data):
    """Prepare structured context for AI assistant with source citations."""
    context = {
        "variant_identification": {
            "source": "ClinGen Allele Registry",
            "data": {
                "caid": clingen_data.get('CAid', 'N/A'),
                "rsid": f"rs{clingen_data.get('rsid')}" if clingen_data.get('rsid') else 'N/A',
                "genomic_hgvs_grch38": clingen_data.get('genomic_hgvs_grch38', 'N/A'),
                "mane_transcript": clingen_data.get('mane_ensembl', 'N/A')
            }
        },
        "functional_predictions": {
            "source": "Ensembl VEP & dbNSFP (via MyVariant.info)",
            "data": {}
        },
        "clinical_significance": {
            "source": "ClinVar (via MyVariant.info)",
            "data": {}
        },
        "population_frequencies": {
            "source": "gnomAD, 1000 Genomes, ExAC (via MyVariant.info)",
            "data": population_freq_data
        }
    }
    
    # Extract VEP predictions
    if vep_data and len(vep_data) > 0:
        transcript_consequences = vep_data[0].get('transcript_consequences', [])
        if transcript_consequences:
            primary = transcript_consequences[0]
            context["functional_predictions"]["data"] = {
                "gene": primary.get('gene_symbol', 'N/A'),
                "transcript": primary.get('transcript_id', 'N/A'),
                "consequence": ', '.join(primary.get('consequence_terms', [])),
                "impact": primary.get('impact', 'N/A'),
                "amino_acid_change": primary.get('amino_acids', 'N/A'),
                "sift_score": primary.get('sift_score'),
                "sift_prediction": primary.get('sift_prediction'),
                "polyphen_score": primary.get('polyphen_score'),
                "polyphen_prediction": primary.get('polyphen_prediction')
            }
    
    # Extract clinical significance
    if isinstance(myvariant_data, dict):
        clinvar = myvariant_data.get('clinvar', {})
        if clinvar:
            context["clinical_significance"]["data"] = {
                "significance": clinvar.get('clinical_significance', 'N/A'),
                "variant_id": clinvar.get('variant_id', 'N/A'),
                "review_status": 'N/A'
            }
            
            rcv_data = clinvar.get('rcv', [])
            if rcv_data and isinstance(rcv_data, list) and len(rcv_data) > 0:
                context["clinical_significance"]["data"]["review_status"] = rcv_data[0].get('review_status', 'N/A')
        
        # Add comprehensive prediction scores
        dbnsfp = myvariant_data.get('dbnsfp', {})
        if dbnsfp:
            prediction_scores = {}
            
            # Key predictors for AI analysis
            predictors = {
                'CADD': ['cadd', 'phred'],
                'REVEL': ['revel', 'score'],
                'MetaSVM': ['metasvm', 'score'],
                'MetaSVM_pred': ['metasvm', 'pred'],
                'BayesDel': ['bayesdel', 'add_af', 'score'],
                'ClinPred': ['clinpred', 'score'],
                'AlphaMissense': ['alphamissense', 'score'],
                'GERP++': ['gerp++', 'rs']
            }
            
            def extract_nested(data, path):
                current = data
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return None
                if isinstance(current, list) and current:
                    return current[0]
                return current
            
            for name, path in predictors.items():
                value = extract_nested(dbnsfp, path)
                if value is not None:
                    prediction_scores[name] = value
            
            if prediction_scores:
                context["functional_predictions"]["data"]["additional_scores"] = prediction_scores
    
    return context

def query_openai_assistant(context_data, user_question, api_key):
    """Query OpenAI with structured context and ensure proper citations."""
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Build the system prompt with citation requirements
    system_prompt = """You are a clinical genetics expert assistant analyzing variant data. Your role is to:

1. Provide accurate, evidence-based interpretations of genetic variant data
2. ALWAYS cite the specific data source for every claim you make
3. Use the format [Source: Database/Tool] after each statement
4. Integrate population-specific frequency data when relevant
5. Explain the clinical significance in clear, accessible language
6. Highlight any conflicts or uncertainties in the data

CRITICAL CITATION RULES:
- Every factual claim must be followed by [Source: X]
- Available sources: ClinGen Allele Registry, Ensembl VEP, MyVariant.info, ClinVar, gnomAD, 1000 Genomes, ExAC, dbNSFP
- Example: "This variant is classified as pathogenic [Source: ClinVar]"
- If data is missing or unavailable, state this explicitly

Be concise but thorough. Prioritize clinical actionability."""

    # Structure the variant data for the prompt
    variant_info = f"""
VARIANT IDENTIFICATION:
{json.dumps(context_data['variant_identification'], indent=2)}

FUNCTIONAL PREDICTIONS:
{json.dumps(context_data['functional_predictions'], indent=2)}

CLINICAL SIGNIFICANCE:
{json.dumps(context_data['clinical_significance'], indent=2)}

POPULATION FREQUENCIES (User-selected populations):
{json.dumps(context_data['population_frequencies'], indent=2)}
"""

    user_prompt = f"""Analyze this genetic variant data:

{variant_info}

User Question: {user_question}

Provide a comprehensive analysis with proper citations for every claim."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4 for best accuracy
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error querying OpenAI: {str(e)}"

def display_population_frequencies(population_freq_data):
    """Display filtered population frequency data."""
    st.subheader("📊 Population Frequency Analysis (Filtered)")
    
    if not any(population_freq_data.values()):
        st.info("No population frequency data available for selected populations")
        return
    
    # Create tabs for different databases
    freq_tabs = st.tabs(["gnomAD Exome", "gnomAD Genome", "1000 Genomes", "ExAC"])
    
    with freq_tabs[0]:
        if population_freq_data['gnomad_exome']:
            st.markdown("**gnomAD Exome v2.1.1**")
            df_data = []
            for pop, data in population_freq_data['gnomad_exome'].items():
                df_data.append({
                    'Population': pop,
                    'Frequency': f"{data['frequency']:.6f}",
                    'Allele Count': data['allele_count'],
                    'Allele Number': data['allele_number']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                chart_data = pd.DataFrame(df_data)
                chart_data['Frequency'] = chart_data['Frequency'].astype(float)
                st.bar_chart(chart_data.set_index('Population')['Frequency'])
        else:
            st.info("No gnomAD exome data for selected populations")
    
    with freq_tabs[1]:
        if population_freq_data['gnomad_genome']:
            st.markdown("**gnomAD Genome v3.1.2**")
            df_data = []
            for pop, data in population_freq_data['gnomad_genome'].items():
                df_data.append({
                    'Population': pop,
                    'Frequency': f"{data['frequency']:.6f}",
                    'Allele Count': data['allele_count'],
                    'Allele Number': data['allele_number']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                chart_data = pd.DataFrame(df_data)
                chart_data['Frequency'] = chart_data['Frequency'].astype(float)
                st.bar_chart(chart_data.set_index('Population')['Frequency'])
        else:
            st.info("No gnomAD genome data for selected populations")
    
    with freq_tabs[2]:
        if population_freq_data['1000genomes']:
            st.markdown("**1000 Genomes Project Phase 3**")
            df_data = []
            for pop, data in population_freq_data['1000genomes'].items():
                df_data.append({
                    'Population': pop,
                    'Frequency': f"{data['frequency']:.6f}",
                    'Allele Count': data['allele_count']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No 1000 Genomes data for selected populations")
    
    with freq_tabs[3]:
        if population_freq_data['exac']:
            st.markdown("**Exome Aggregation Consortium (ExAC)**")
            df_data = []
            for pop, data in population_freq_data['exac'].items():
                df_data.append({
                    'Population': pop,
                    'Frequency': f"{data['frequency']:.6f}",
                    'Allele Count': data['allele_count']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No ExAC data for selected populations")

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Genetic Variant Analyzer with AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 OpenAI API Configuration")
        
        # Try to get API key from environment variable first (repository secret)
        api_key = os.environ.get('OPEN_AI_API_KEY', '')
        
        if not api_key:
            # Fallback to manual input if environment variable not found
            api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI analysis")
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API key not found. Please set OPEN_AI_API_KEY environment variable or enter manually.")
        
        st.markdown("---")
        
        st.markdown("### 👥 Population Selection")
        st.write("Select populations to focus your analysis:")
        
        available_populations = [
            'All Populations',
            'African/African American',
            'Latino/Admixed American',
            'Ashkenazi Jewish',
            'East Asian',
            'Finnish',
            'Non-Finnish European',
            'South Asian',
            'Middle Eastern',
            'Amish',
            'Other',
            'Overall'
        ]
        
        selected_populations = st.multiselect(
            "Choose populations:",
            options=available_populations,
            default=['Overall', 'South Asian'],  # Default relevant to user location
            help="Select specific populations for frequency analysis. 'All Populations' will show all available data."
        )
        
        if not selected_populations:
            st.warning("⚠️ Please select at least one population")
        
        st.markdown("---")
        
        st.markdown("### About")
        st.write("""
        This tool analyzes genetic variants using multiple genomic databases and AI-powered interpretation.
        
        **Data Sources:**
        - ClinGen Allele Registry
        - MyVariant.info
        - Ensembl VEP
        - gnomAD
        - 1000 Genomes
        - ClinVar
        """)
        
        st.markdown("### Example Variants")
        if st.button("Load Example: NDUFS8", key="example1"):
            st.session_state.example_input = "NM_002496.3:c.64C>T"
        if st.button("Load Example: BRCA1", key="example2"):
            st.session_state.example_input = "NM_007294.3:c.5266dupC"
        if st.button("Load Example: RSID", key="example3"):
            st.session_state.example_input = "rs369602258"
    
    # Main input
    st.markdown('<div class="section-header">Variant Input</div>', unsafe_allow_html=True)
    
    default_value = getattr(st.session_state, 'example_input', "")
    user_input = st.text_input(
        "Enter a genetic variant (HGVS notation or RSID):",
        value=default_value,
        placeholder="e.g., NM_002496.3:c.64C>T or rs369602258",
        key="variant_input"
    )
    
    # Clear the example input after it's been used
    if hasattr(st.session_state, 'example_input'):
        delattr(st.session_state, 'example_input')
    
    analyze_button = st.button("🔬 Analyze Variant", type="primary", key="analyze_btn")
    
    # Check if we should show results
    should_analyze = analyze_button and user_input
    should_show_results = False
    
    if should_analyze:
        # Validate population selection
        if not selected_populations:
            st.error("Please select at least one population from the sidebar")
            st.stop()
        
        # Store the analysis in session state
        if 'analysis_data' not in st.session_state or st.session_state.get('last_query') != user_input:
            with st.spinner("Analyzing variant..."):
                # Initialize router
                router = GenomicQueryRouter()
                classification = router.classify_query(user_input)
                
                if not classification.is_genomic:
                    st.error("Invalid input format. Please provide a valid HGVS notation or RSID.")
                    st.stop()
                
                try:
                    start_time = time.time()
                    
                    # Handle different query types
                    if classification.query_type == 'rsid':
                        st.info("🔍 RSID detected - querying MyVariant.info and Ensembl VEP directly (skipping ClinGen)")
                        clingen_data = {
                            'CAid': 'N/A (RSID input)',
                            'rsid': classification.extracted_identifier.replace('rs', ''),
                            'genomic_hgvs_grch38': None,
                            'genomic_hgvs_grch37': None,
                            'myvariant_hg38': None,
                            'myvariant_hg19': None,
                            'mane_ensembl': None,
                            'mane_refseq': None
                        }
                        annotations = get_variant_annotations(clingen_data, classification)
                        
                        if annotations['myvariant_data']:
                            myv_data = annotations['myvariant_data']
                            
                            if isinstance(myv_data, list) and len(myv_data) > 0:
                                myv_data = myv_data[0]
                                annotations['myvariant_data'] = myv_data
                            
                            if isinstance(myv_data, dict):
                                clingen_info = myv_data.get('clingen', {})
                                if clingen_info.get('caid'):
                                    clingen_data['CAid'] = clingen_info['caid']
                                
                                clinvar_info = myv_data.get('clinvar', {})
                                if clinvar_info.get('hgvs'):
                                    hgvs_data = clinvar_info['hgvs']
                                    if isinstance(hgvs_data, dict):
                                        if hgvs_data.get('coding'):
                                            coding_hgvs = hgvs_data['coding']
                                            try:
                                                vep_url = f"https://rest.ensembl.org/vep/human/hgvs/{coding_hgvs}"
                                                vep_headers = {"Content-Type": "application/json", "Accept": "application/json"}
                                                vep_response = requests.get(vep_url, headers=vep_headers, timeout=30)
                                                if vep_response.ok:
                                                    annotations['vep_data'] = vep_response.json()
                                            except:
                                                pass
                    else:
                        # For HGVS notations, query ClinGen first
                        clingen_raw = query_clingen_allele(classification.extracted_identifier)
                        clingen_data = parse_caid_minimal(clingen_raw)
                        annotations = get_variant_annotations(clingen_data, classification)
                    
                    # Extract population frequencies based on user selection
                    population_freq_data = extract_population_frequencies(
                        annotations['myvariant_data'],
                        selected_populations
                    )
                    
                    # Prepare AI context
                    ai_context = prepare_ai_context(
                        clingen_data,
                        annotations['vep_data'],
                        annotations['myvariant_data'],
                        population_freq_data
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Store in session state
                    st.session_state.analysis_data = {
                        'classification': classification,
                        'clingen_data': clingen_data,
                        'annotations': annotations,
                        'population_freq_data': population_freq_data,
                        'ai_context': ai_context,
                        'processing_time': processing_time,
                        'selected_populations': selected_populations
                    }
                    st.session_state.last_query = user_input
                    should_show_results = True
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
                    st.stop()
        else:
            should_show_results = True
    
    # Show results if we have analysis data in session state
    elif 'analysis_data' in st.session_state and st.session_state.get('last_query'):
        should_show_results = True
    
    # Display results section - ALL ORIGINAL FUNCTIONALITIES PRESERVED
    if should_show_results and 'analysis_data' in st.session_state:
        # Retrieve analysis data from session state
        analysis_data = st.session_state.analysis_data
        classification = analysis_data['classification']
        clingen_data = analysis_data['clingen_data']
        annotations = analysis_data['annotations']
        population_freq_data = analysis_data['population_freq_data']
        ai_context = analysis_data['ai_context']
        processing_time = analysis_data['processing_time']
        stored_populations = analysis_data.get('selected_populations', [])
        
        # Show clear results button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Results", key="clear_results"):
                if 'analysis_data' in st.session_state:
                    del st.session_state['analysis_data']
                if 'last_query' in st.session_state:
                    del st.session_state['last_query']
                st.rerun()
        
        with col2:
            st.markdown(f"**Analyzing:** {classification.extracted_identifier}")
        
        # Display classification info
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Detected identifier:** {classification.extracted_identifier}")
        with col2:
            st.write(f"**Type:** {classification.query_type}")
        with col3:
            st.write(f"**Selected populations:** {len(stored_populations)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display ClinGen results
        st.markdown('<div class="section-header">ClinGen Allele Registry</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**CAid:** {clingen_data.get('CAid', 'N/A')}")
            st.write(f"**RSID:** {clingen_data.get('rsid', 'N/A')}")
        with col2:
            mane_ensembl = clingen_data.get('mane_ensembl', 'N/A')
            myvariant_id = clingen_data.get('myvariant_hg38', 'N/A')
            
            st.write(f"**MANE Ensembl:** {mane_ensembl}")
            st.write(f"**MyVariant ID:** {myvariant_id}")
        
        # Display any API errors
        if annotations['errors']:
            for error in annotations['errors']:
                st.warning(f"⚠️ {error}")
        
        # NEW: AI Assistant Section (if API key is available)
        if api_key:
            st.markdown('<div class="section-header">🤖 AI-Powered Analysis</div>', unsafe_allow_html=True)
            
            with st.expander("💬 Ask AI Assistant", expanded=True):
                st.write("Ask questions about this variant. The AI will analyze the data with proper citations.")
                
                # Predefined questions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 Summarize Clinical Significance", key="ai_q1"):
                        st.session_state.ai_question = "Provide a comprehensive summary of the clinical significance of this variant, including pathogenicity predictions and population frequency context."
                with col2:
                    if st.button("🧬 Explain Functional Impact", key="ai_q2"):
                        st.session_state.ai_question = "Explain the functional impact of this variant at the protein level and its predicted pathogenicity."
                with col3:
                    if st.button("👥 Population Analysis", key="ai_q3"):
                        st.session_state.ai_question = "Analyze the population frequency data for this variant and discuss its clinical implications in the selected populations."
                
                # Custom question input
                default_question = getattr(st.session_state, 'ai_question', '')
                user_question = st.text_area(
                    "Or ask your own question:",
                    value=default_question,
                    placeholder="e.g., What is the clinical significance of this variant in South Asian populations?",
                    key="custom_ai_question"
                )
                
                if hasattr(st.session_state, 'ai_question'):
                    delattr(st.session_state, 'ai_question')
                
                if st.button("🚀 Get AI Analysis", type="primary", key="query_ai"):
                    if user_question:
                        with st.spinner("🤖 AI is analyzing the variant data..."):
                            ai_response = query_openai_assistant(ai_context, user_question, api_key)
                            
                            st.markdown('<div class="ai-response-box">', unsafe_allow_html=True)
                            st.markdown("### AI Analysis")
                            st.markdown(ai_response)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add disclaimer
                            st.caption("⚠️ AI-generated analysis for research purposes only. Verify all information with source databases and consult clinical geneticists for medical decisions.")
                    else:
                        st.warning("Please enter a question or select a predefined question.")
        
        # Continue with ALL ORIGINAL TABS AND FUNCTIONALITY
        # Main analysis tabs - COMPLETE ORIGINAL IMPLEMENTATION
        if annotations['myvariant_data'] or annotations['vep_data']:
            st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
            
            # Add population frequency tab to the original tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🧬 VEP Analysis", 
                "🔬 MyVariant Analysis", 
                "👥 Population Frequencies",
                "🏥 Clinical Data", 
                "📋 Raw Data",
                "📥 Downloads"
            ])
            
            with tab1:  # VEP Analysis - ORIGINAL CODE
                if annotations['vep_data']:
                    # Use original display_vep_analysis function
                    # This function is fully preserved from original code
                    display_vep_analysis_original(annotations['vep_data'])
                else:
                    st.info("No VEP data available. This may be due to API limitations or the variant not being found in Ensembl.")
            
            with tab2:  # MyVariant Analysis - ORIGINAL CODE
                if annotations['myvariant_data']:
                    # Use original comprehensive display function
                    display_comprehensive_myvariant_data_original(annotations['myvariant_data'])
                else:
                    st.info("No MyVariant data available")
            
            with tab3:  # NEW: Filtered Population Frequencies
                display_population_frequencies(population_freq_data)
                
                # Show which populations were selected
                st.info(f"📍 Showing data for: {', '.join(stored_populations)}")
            
            with tab4:  # Clinical Data - ORIGINAL CODE  
                if annotations['myvariant_data']:
                    myvariant_data = annotations['myvariant_data']
                    
                    # Clinical significance from ClinVar
                    clinvar_data = myvariant_data.get('clinvar', {})
                    if clinvar_data:
                        st.subheader("ClinVar Clinical Significance")
                        
                        clinical_sig = (clinvar_data.get('clinical_significance') or 
                                       clinvar_data.get('clnsig') or 'N/A')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Clinical Significance", clinical_sig)
                        with col2:
                            if clinvar_data.get('variant_id'):
                                st.metric("ClinVar ID", clinvar_data['variant_id'])
                        with col3:
                            if clinvar_data.get('allele_id'):
                                st.metric("Allele ID", clinvar_data['allele_id'])
                        
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
                                            if isinstance(conditions, dict) and conditions.get('name'):
                                                st.write(f"**Associated Condition:** {conditions['name']}")
                    
                    # UniProt clinical data
                    uniprot_data = myvariant_data.get('uniprot', {})
                    if uniprot_data and uniprot_data.get('clinical_significance'):
                        st.subheader("UniProt Clinical Annotation")
                        st.write(f"**Clinical Significance:** {uniprot_data['clinical_significance']}")
                        if uniprot_data.get('source_db_id'):
                            st.write(f"**Source:** {uniprot_data['source_db_id']}")
                    
                    # Population frequency summary for clinical context
                    st.subheader("Population Frequency Context")
                    
                    max_freq = 0
                    freq_source = "N/A"
                    
                    gnomad_exome = myvariant_data.get('gnomad_exome', {})
                    if gnomad_exome and gnomad_exome.get('af', {}).get('af'):
                        exome_freq = gnomad_exome['af']['af']
                        if exome_freq > max_freq:
                            max_freq = exome_freq
                            freq_source = "gnomAD Exome"
                    
                    gnomad_genome = myvariant_data.get('gnomad_genome', {})
                    if gnomad_genome and gnomad_genome.get('af', {}).get('af'):
                        genome_freq = gnomad_genome['af']['af']
                        if genome_freq > max_freq:
                            max_freq = genome_freq
                            freq_source = "gnomAD Genome"
                    
                    if max_freq > 0:
                        st.metric(f"Max Population Frequency ({freq_source})", f"{max_freq:.6f}")
                        
                        if max_freq >= 0.01:
                            st.success("🟢 Common variant (≥1%)")
                        elif max_freq >= 0.005:
                            st.warning("🟡 Low frequency variant (0.5-1%)")
                        elif max_freq >= 0.0001:
                            st.info("🔵 Rare variant (0.01-0.5%)")
                        else:
                            st.error("🔴 Very rare variant (<0.01%)")
                    else:
                        st.info("No reliable population frequency data available")
                
                else:
                    st.info("No clinical data available")
            
            with tab5:  # Raw Data - ORIGINAL CODE
                st.subheader("Raw API Responses")
                
                with st.expander("ClinGen Allele Registry Data", expanded=False):
                    st.json(clingen_data)
                
                if annotations['myvariant_data']:
                    with st.expander("MyVariant.info Data", expanded=False):
                        st.json(annotations['myvariant_data'])
                
                if annotations['vep_data']:
                    with st.expander("Ensembl VEP Data", expanded=False):
                        st.json(annotations['vep_data'])
                
                # Show AI context that was sent to OpenAI
                if api_key:
                    with st.expander("AI Context Data (Sent to OpenAI)", expanded=False):
                        st.json(ai_context)
            
            with tab6:  # Downloads - ORIGINAL CODE with population data addition
                st.subheader("📥 Download Data")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if clingen_data:
                        clingen_json = json.dumps(clingen_data, indent=2)
                        st.download_button(
                            label="📋 ClinGen Data",
                            data=clingen_json,
                            file_name=f"clingen_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json",
                            mime="application/json",
                            key=f"clingen_dl_{classification.extracted_identifier}",
                            help="Download ClinGen Allele Registry data as JSON"
                        )
                
                with col2:
                    if annotations['myvariant_data']:
                        myvariant_json = json.dumps(annotations['myvariant_data'], indent=2)
                        st.download_button(
                            label="🔬 MyVariant Data", 
                            data=myvariant_json,
                            file_name=f"myvariant_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json",
                            mime="application/json",
                            key=f"myvariant_dl_{classification.extracted_identifier}",
                            help="Download MyVariant.info annotations as JSON"
                        )
                
                with col3:
                    if annotations['vep_data']:
                        vep_json = json.dumps(annotations['vep_data'], indent=2)
                        st.download_button(
                            label="🧬 VEP Data",
                            data=vep_json, 
                            file_name=f"vep_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json",
                            mime="application/json",
                            key=f"vep_dl_{classification.extracted_identifier}",
                            help="Download Ensembl VEP predictions as JSON"
                        )
                
                with col4:
                    # NEW: Download filtered population frequency data
                    if population_freq_data:
                        pop_freq_json = json.dumps(population_freq_data, indent=2)
                        st.download_button(
                            label="👥 Population Data",
                            data=pop_freq_json,
                            file_name=f"population_freq_{classification.extracted_identifier.replace(':', '_').replace('>', '_')}.json",
                            mime="application/json",
                            key=f"popfreq_dl_{classification.extracted_identifier}",
                            help="Download filtered population frequency data as JSON"
                        )
        
        # Processing time and summary
        st.success(f"✅ Analysis completed in {processing_time:.2f} seconds")
        
        # Analysis summary
        with st.expander("Analysis Summary", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Sources Retrieved:**")
                sources = ["ClinGen Allele Registry"]
                if annotations['myvariant_data']:
                    sources.append("MyVariant.info")
                if annotations['vep_data']:
                    sources.append("Ensembl VEP")
                for source in sources:
                    st.write(f"• {source}")
            
            with col2:
                st.write("**Key Identifiers:**")
                if clingen_data.get('CAid'):
                    st.write(f"• ClinGen CAID: {clingen_data['CAid']}")
                if clingen_data.get('rsid'):
                    st.write(f"• dbSNP RSID: rs{clingen_data['rsid']}")
                if annotations['myvariant_data'] and annotations['myvariant_data'].get('clinvar', {}).get('variant_id'):
                    st.write(f"• ClinVar ID: {annotations['myvariant_data']['clinvar']['variant_id']}")
    
    elif user_input and not analyze_button:
        # Show input validation without analyzing
        router = GenomicQueryRouter()
        classification = router.classify_query(user_input)
        
        if classification.is_genomic:
            st.success(f"✅ Valid {classification.query_type} format detected: {classification.extracted_identifier}")
        else:
            st.error("❌ Invalid format. Please provide a valid HGVS notation or RSID.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🧬 <strong>Genetic Variant Analyzer with AI Assistant</strong></p>
        <p>Data sources: ClinGen Allele Registry • MyVariant.info • Ensembl VEP • OpenAI GPT-4</p>
        <p>For research purposes only • Not for clinical use</p>
    </div>
    """, unsafe_allow_html=True)

# Add the original display functions here - these are COMPLETE and UNTRUNCATED
def display_vep_analysis_original(vep_data):
    """Display comprehensive VEP analysis - ORIGINAL FUNCTION."""
    if not vep_data or not vep_data[0].get('transcript_consequences'):
        st.warning("No VEP data available")
        return
    
    variant_info = vep_data[0]
    all_transcripts = variant_info.get('transcript_consequences', [])
