import streamlit as st
import pandas as pd
import numpy as np

# --- Funzioni Core (OTTIMIZZATE PER VELOCITÀ) ---

def find_and_calculate_matches(reference_df, target_df, mz_tolerance, rt_tolerance):
    """
    Trova e calcola i candidati usando operazioni vettoriali ottimizzate.
    Conserva tutte le colonne originali dei file di input.
    """
    # 1. Validazione e preparazione dei dati
    ref_df = reference_df.copy()
    tgt_df = target_df.copy()
    
    for df in [ref_df, tgt_df]:
        for col in ['m/z', 'RT']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    ref_df.dropna(subset=['m/z', 'RT'], inplace=True)
    tgt_df.dropna(subset=['m/z', 'RT'], inplace=True)
    
    # Reset index per target per avere target_index corretto
    tgt_df = tgt_df.reset_index(drop=False).rename(columns={'index': 'target_index'})
    
    # --- MODIFICA: Rilevamento dinamico della colonna AREA ---
    area_col_name = 'AREA MEDIA' if 'AREA MEDIA' in ref_df.columns and 'AREA MEDIA' in tgt_df.columns else 'AREA'
    has_area_col = area_col_name in ref_df.columns and area_col_name in tgt_df.columns

    # 2. Creazione di un cross-join che mantiene tutte le colonne
    ref_df['key'] = 1
    tgt_df['key'] = 1
    merged = pd.merge(ref_df, tgt_df, on='key', suffixes=('_ref', '_tgt')).drop('key', axis=1)
    
    # 3. Calcolo vettoriale dei delta
    merged['mz_diff'] = np.abs(merged['m/z_ref'] - merged['m/z_tgt'])
    merged['rt_diff'] = np.abs(merged['RT_ref'] - merged['RT_tgt'])
    
    # 4. Filtro vettoriale
    mask = (merged['mz_diff'] <= mz_tolerance) & (merged['rt_diff'] <= rt_tolerance)
    matches = merged[mask].copy()
    
    if matches.empty:
        return pd.DataFrame()
    
    # 5. Aggiunta di colonne calcolate al dataframe dei match
    matches['is_exact_mz'] = np.isclose(matches['m/z_ref'], matches['m/z_tgt'])
    
    if has_area_col:
        matches['area_diff'] = np.abs(matches[f'{area_col_name}_tgt'].fillna(0) - matches[f'{area_col_name}_ref'].fillna(0))
    
    # 6. Rinomina delle colonne chiave per coerenza
    matches = matches.rename(columns={
        'm/z_ref': 'ref_mz',
        'RT_ref': 'ref_RT',
        'm/z_tgt': 'target_mz',
        'RT_tgt': 'target_RT',
    })
    
    if 'VarName_ref' in matches.columns:
        matches = matches.rename(columns={'VarName_ref': 'VarName'})
            
    return matches


def process_assignments(all_matches):
    """
    Separa le assegnazioni automatiche dai conflitti.
    """
    if all_matches.empty:
        return pd.DataFrame(), pd.DataFrame()

    auto_assignments_list = []
    conflicts_list = []
    
    for var_name, var_matches in all_matches.groupby('VarName'):
        exact_matches = var_matches[var_matches['is_exact_mz'] == True]
        potential_assignments = exact_matches if not exact_matches.empty else var_matches
        
        if len(potential_assignments) == 1:
            auto_assignments_list.append(potential_assignments)
        elif len(potential_assignments) > 1:
            conflicts_list.append(potential_assignments)

    auto_assignments = pd.concat(auto_assignments_list, ignore_index=True) if auto_assignments_list else pd.DataFrame()
    conflicts = pd.concat(conflicts_list, ignore_index=True) if conflicts_list else pd.DataFrame()
    
    return auto_assignments, conflicts


def validate_dataframe(df, required_columns, sheet_name):
    """Valida che il DataFrame abbia le colonne minime richieste."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Colonne obbligatorie mancanti in {sheet_name}: {missing_cols}")
        st.info(f"Colonne disponibili: {list(df.columns)}")
        return False
    return True

# --- NUOVA FUNZIONE HELPER ---
def deduplicate_columns(df):
    """
    Rinomina le colonne duplicate in un DataFrame aggiungendo un suffisso.
    Es: ['A', 'B', 'A'] -> ['A', 'B', 'A_1']
    Gestisce anche i nomi di colonna numerici convertendoli in stringhe.
    """
    new_cols = []
    col_counts = {}
    for col in df.columns:
        col_str = str(col) # Converte in stringa per sicurezza
        if col_str in col_counts:
            col_counts[col_str] += 1
            new_cols.append(f"{col_str}_{col_counts[col_str]}")
        else:
            col_counts[col_str] = 0
            new_cols.append(col_str)
    df.columns = new_cols
    return df

# --- Inizializzazione Session State ---
def initialize_session_state():
    """Inizializza le variabili di sessione."""
    defaults = {
        'step': 'initial',
        'conflicts': pd.DataFrame(),
        'auto_assignments': pd.DataFrame(),
        'all_assignments': pd.DataFrame(),
        'original_target_df': pd.DataFrame(),
        'original_ref_df': pd.DataFrame()
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# --- Interfaccia Utente ---
st.set_page_config(layout="wide", page_title="LC-Orbitrap Aligner")
st.title("🔬 LC-Orbitrap Aligner")

initialize_session_state()

# --- Sidebar di Configurazione ---
with st.sidebar:
    st.header("⚙️ Impostazioni")
    uploaded_file = st.file_uploader("1. Carica il tuo file Excel", type=['xlsx'])
    
    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            st.info(f"Fogli trovati: {', '.join(sheet_names)}")
            
            reference_sheet = st.selectbox("2. Seleziona il foglio di Riferimento", sheet_names, index=0)
            target_sheet = st.selectbox("3. Seleziona il foglio Target", sheet_names, 
                                       index=1 if len(sheet_names) > 1 else 0)
        except Exception as e:
            st.error(f"Errore nella lettura del file Excel: {str(e)}")
            uploaded_file = None

    st.subheader("Imposta Tolleranze")
    mz_tolerance = st.number_input("Tolleranza m/z (±)", 0.0001, 1.0, 0.001, 0.0001, format="%.4f")
    rt_tolerance = st.number_input("Tolleranza RT (± minuti)", 0.01, 10.0, 0.5, 0.01, format="%.2f")
    start_button = st.button("Avvia Allineamento", type="primary")

# --- Logica Principale ---
if start_button and uploaded_file:
    if reference_sheet == target_sheet:
        st.warning("⚠️ Seleziona due fogli diversi.")
    else:
        with st.spinner("Allineamento in corso..."):
            try:
                ref_df = pd.read_excel(uploaded_file, sheet_name=reference_sheet)
                target_df = pd.read_excel(uploaded_file, sheet_name=target_sheet)
                
                # FIX 1: Rimuove colonne senza nome o NaN dall'intestazione.
                ref_df = ref_df.loc[:, ref_df.columns.notna()]
                target_df = target_df.loc[:, target_df.columns.notna()]

                # FIX 2: Rinomina colonne duplicate (es. ID campioni uguali).
                ref_df = deduplicate_columns(ref_df)
                target_df = deduplicate_columns(target_df)

                ref_df.columns = ref_df.columns.str.strip()
                target_df.columns = target_df.columns.str.strip()
                if 'RT [min]' in target_df.columns:
                    target_df = target_df.rename(columns={'RT [min]': 'RT'})
                if 'RT [min]' in ref_df.columns:
                    ref_df = ref_df.rename(columns={'RT [min]': 'RT'})
                
                if not validate_dataframe(ref_df, ['VarName', 'm/z', 'RT'], "Riferimento") or \
                   not validate_dataframe(target_df, ['m/z', 'RT'], "Target"):
                    st.stop()

                st.session_state.original_target_df = target_df.copy()
                st.session_state.original_ref_df = ref_df.copy()
                
                all_matches = find_and_calculate_matches(ref_df, target_df, mz_tolerance, rt_tolerance)
                
                if all_matches.empty:
                    st.session_state.step = 'no_matches'
                else:
                    auto_assignments, conflicts = process_assignments(all_matches)
                    st.session_state.auto_assignments = auto_assignments
                    st.session_state.conflicts = conflicts
                    
                    if conflicts.empty:
                        st.session_state.step = 'final_results_auto'
                    else:
                        st.session_state.step = 'conflict_resolution'
                st.rerun()
                
            except Exception as e:
                st.error(f"Errore durante l'allineamento: {str(e)}")
                st.exception(e)

# --- Risoluzione Conflitti ---
if st.session_state.step == 'conflict_resolution':
    st.subheader("Riepilogo e Risoluzione Manuale")
    st.markdown("---")
    
    if not st.session_state.auto_assignments.empty:
        st.info("Le seguenti variabili sono state assegnate **automaticamente**.")
        with st.expander("✅ Visualizza Assegnazioni Automatiche", expanded=True):
            display_cols_auto = ['VarName', 'ref_mz', 'ref_RT', 'target_mz', 'target_RT', 'mz_diff', 'rt_diff']
            if 'area_diff' in st.session_state.auto_assignments.columns:
                display_cols_auto.append('area_diff')
            st.dataframe(st.session_state.auto_assignments[display_cols_auto], use_container_width=True)
    
    all_ref_vars = set(st.session_state.original_ref_df['VarName'])
    auto_assigned_vars = set()
    if not st.session_state.auto_assignments.empty: 
        auto_assigned_vars = set(st.session_state.auto_assignments['VarName'])
    conflicted_vars_set = set(st.session_state.conflicts['VarName'])
    unassigned_vars = all_ref_vars - auto_assigned_vars - conflicted_vars_set
    
    if unassigned_vars:
        st.warning(f"**Nessun candidato trovato** per {len(unassigned_vars)} variabili.")
        with st.expander("❌ Visualizza Variabili Non Assegnate"):
            unassigned_df = st.session_state.original_ref_df[
                st.session_state.original_ref_df['VarName'].isin(list(unassigned_vars))
            ]
            st.dataframe(unassigned_df, use_container_width=True)
    
    st.markdown("---")
    
    with st.form("conflict_form"):
        resolutions = {}
        conflict_groups = st.session_state.conflicts.groupby(['VarName', 'ref_mz', 'ref_RT'])
        
        st.info(f"Risolvi i **conflitti** per le {len(conflict_groups)} variabili rimanenti. Il candidato migliore è preselezionato.")

        for (var, ref_mz, ref_rt), group in conflict_groups:
            ref_info = f"**Variabile:** `{var}` (ref m/z: **{ref_mz:.4f}** | ref RT: **{ref_rt:.3f}**)"
            sort_columns = ['mz_diff', 'rt_diff']
            if 'area_diff' in group.columns:
                sort_columns.append('area_diff')
            group = group.sort_values(sort_columns)
        
            options = []
            for _, r in group.iterrows():
                area_info = ""
                if 'area_diff' in r and pd.notna(r['area_diff']):
                    area_info = f" | Area Δ: {r['area_diff']:,.0f}".replace(",", ".")
                
                label = (f"m/z: {r['target_mz']:.4f} (Δ: {r['mz_diff']:.4f}) | "
                         f"RT: {r['target_RT']:.3f} (Δ: {r['rt_diff']:.3f})"
                         f"{area_info}")
                options.append({'label': label, 'value': int(r['target_index'])})
            
            options.append({'label': "Nessuno di questi - non assegnare", 'value': -1})
            
            option_values = [opt['value'] for opt in options]
            option_labels = [opt['label'] for opt in options]
            widget_key = f"{var}_{ref_mz}_{ref_rt}"
            
            resolutions[widget_key] = {
                'selection': st.radio(
                    ref_info, 
                    option_values, 
                    index=0,
                    format_func=lambda v: option_labels[option_values.index(v)], 
                    key=widget_key
                ),
                'var_info': (var, ref_mz, ref_rt)
            }

        if st.form_submit_button("Conferma e Genera Risultati Finali"):
            resolved_assignments_list = []
            for res_key, resolution in resolutions.items():
                target_idx = resolution['selection']
                
                if target_idx != -1:
                    var, ref_mz, ref_rt = resolution['var_info']
                    selected_match = st.session_state.conflicts[
                        (st.session_state.conflicts['VarName'] == var) &
                        (np.isclose(st.session_state.conflicts['ref_mz'], ref_mz)) &
                        (np.isclose(st.session_state.conflicts['ref_RT'], ref_rt)) &
                        (st.session_state.conflicts['target_index'] == target_idx)
                    ].copy()

                    if not selected_match.empty:
                        selected_match['Tipo di Assegnazione'] = 'Manuale'
                        resolved_assignments_list.append(selected_match)

            auto_assignments_with_type = st.session_state.auto_assignments.copy()
            if not auto_assignments_with_type.empty:
                auto_assignments_with_type['Tipo di Assegnazione'] = 'Automatico'
            
            all_assignments_list = []
            if not auto_assignments_with_type.empty:
                all_assignments_list.append(auto_assignments_with_type)
            if resolved_assignments_list:
                all_assignments_list.extend(resolved_assignments_list)
            
            if all_assignments_list:
                st.session_state.all_assignments = pd.concat(all_assignments_list, ignore_index=True)
            else:
                st.session_state.all_assignments = pd.DataFrame()

            st.session_state.step = 'final_results_manual'
            st.rerun()


# --- Risultati Finali ---
if st.session_state.step in ['final_results_manual', 'final_results_auto', 'no_matches']:
    st.subheader("📊 Risultati Finali dell'Allineamento")
    
    if st.session_state.step == 'final_results_auto':
        st.success("🎉 Allineamento completato! Tutte le corrispondenze sono state assegnate automaticamente.")
        st.session_state.all_assignments = st.session_state.auto_assignments.copy()
        if not st.session_state.all_assignments.empty:
            st.session_state.all_assignments['Tipo di Assegnazione'] = 'Automatico'
    
    elif st.session_state.step == 'final_results_manual':
        st.success("✅ Allineamento completato con la risoluzione manuale dei conflitti.")

    elif st.session_state.step == 'no_matches':
        st.warning("Nessun abbinamento trovato con le tolleranze specificate.")
        st.session_state.all_assignments = pd.DataFrame()

    all_assignments = st.session_state.all_assignments
    original_target_df = st.session_state.original_target_df
    original_ref_df = st.session_state.original_ref_df

    matched_df = pd.DataFrame()
    unmatched_df = pd.DataFrame()
    unmatched_ref_df = pd.DataFrame()
    
    if not all_assignments.empty:
        st.markdown("---")
        st.markdown("### ✅ Target Abbinati")
        
        # --- NUOVA LOGICA DI VISUALIZZAZIONE ROBUSTA (FIXED) ---

        # 1. Inizia con i dati dei match.
        base_df = all_assignments.copy()

        # 2. Definisce le colonne "core" e i loro nomi finali in modo esplicito.
        # Questo previene ambiguità e ridenominazioni che creano duplicati.
        core_cols_map = {
            'VarName': 'VarName_ref',
            'VarName_tgt': 'VarName_tgt',
            'Tipo di Assegnazione': 'Tipo di Assegnazione',
            'ref_mz': 'ref_mz',
            'target_mz': 'target_mz',
            'mz_diff': 'mz_diff',
            'ref_RT': 'ref_RT',
            'target_RT': 'target_RT',
            'rt_diff': 'rt_diff',
            'VIPN_ref': 'VIPN_ref',
            'VIPN_tgt': 'VIPN_target'
        }
        
        # Filtra la mappa per includere solo le colonne che esistono realmente nel DataFrame.
        # Questo rende il codice robusto se colonne opzionali (es. VIPN) mancano.
        final_core_cols_map = {
            source: dest for source, dest in core_cols_map.items() if source in base_df.columns
        }
        
        # Seleziona e rinomina le colonne core in un solo passaggio.
        display_df_core = base_df[list(final_core_cols_map.keys())].rename(columns=final_core_cols_map)
        
        # 3. Identifica tutte le altre colonne (campioni, aree, ecc.).
        # Esclude le colonne "core" già processate e le colonne interne di supporto.
        internal_cols = {'target_index', 'key', 'is_exact_mz', 'area_diff'}
        processed_cols = set(final_core_cols_map.keys()) | internal_cols
        
        other_cols = sorted([col for col in base_df.columns if col not in processed_cols])
        
        # 4. Combina il DataFrame core con le altre colonne.
        # Poiché i set di colonne sono disgiunti, non verranno creati duplicati.
        display_df = pd.concat([display_df_core, base_df[other_cols]], axis=1)

        st.dataframe(display_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### ❌ Target Non Abbinati")
        
        matched_indices = all_assignments['target_index'].unique()
        unmatched_df = original_target_df.loc[~original_target_df.index.isin(matched_indices)]
        st.dataframe(unmatched_df, use_container_width=True)

    else:
        st.markdown("---")
        st.markdown("### ❌ Dati Target Originali (nessun abbinamento)")
        unmatched_df = original_target_df
        st.dataframe(unmatched_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚠️ Variabili di Riferimento Non Abbinate")
    
    all_ref_vars = set(original_ref_df['VarName'])
    matched_ref_vars = set(all_assignments['VarName']) if not all_assignments.empty else set()
    unmatched_ref_vars = all_ref_vars - matched_ref_vars

    if unmatched_ref_vars:
        unmatched_ref_df = original_ref_df[
            original_ref_df['VarName'].isin(list(unmatched_ref_vars))
        ].sort_values(by='VarName')
        st.dataframe(unmatched_ref_df, use_container_width=True)
    else:
        st.success("Tutte le variabili di riferimento sono state abbinate!")

    st.markdown("---")
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Per l'export, usiamo il dataframe pulito e ordinato che mostriamo all'utente
        if 'display_df' in locals() and not display_df.empty:
            display_df.to_excel(writer, sheet_name='Target Abbinati', index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name='Target Abbinati', index=False)
        
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, sheet_name='Target Non Abbinati', index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name='Target Non Abbinati', index=False)

        if not unmatched_ref_df.empty:
            unmatched_ref_df.to_excel(writer, sheet_name='Riferimenti Non Abbinati', index=False)
        else:
            pd.DataFrame().to_excel(writer, sheet_name='Riferimenti Non Abbinati', index=False)
    
    excel_data = output.getvalue()

    st.download_button(
        label="📥 Scarica Tutti i Risultati (Excel)",
        data=excel_data,
        file_name="risultati_allineamento.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    total_target = len(original_target_df)
    matched_count = 0
    if not all_assignments.empty:
        matched_count = len(all_assignments['target_index'].unique())
    
    col1, col2 = st.columns(2)
    col1.metric("Target Totali nel File", f"{total_target}")
    percentage = (matched_count/total_target * 100) if total_target > 0 else 0
    metric_value_str = f"{matched_count} ({percentage:.1f}%)"
    col2.metric("Target Abbinati", metric_value_str)

if st.session_state.step != 'initial':
    if st.button("🔄 Inizia un Nuovo Allineamento"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
