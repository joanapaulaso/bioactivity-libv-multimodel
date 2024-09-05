import os
import streamlit as st
import pandas as pd
from PIL import Image
import altair as alt
from utils.data_search import search_target, select_target
from utils.data_processing import pIC50, norm_value, label_bioactivity, convert_ugml_nm
from utils.descriptors import lipinski, desc_calc
from utils.model import (
    build_model,
    model_generation,
    generate_class_models,
    detect_and_remove_outliers,
    list_models,
    select_algorithms_ui,
)
from utils.visualization import molecules_graph_analysis, mannwhitney
from utils.admet_evaluation import evaluate_admet, summarize_results
from utils.mol_draw import get_molecular_image, image_to_base64
from utils.mol_classification import classify_compound


# Interface principal do aplicativo
def main():
    st.set_page_config(layout="wide")
    image = Image.open("logo.png")
    st.image(image, use_column_width=True)

    st.markdown(
        """
    # Aplicação de Predição de Bioatividade
                
    Essa aplicação permite preparar modelos de Machine Learning utilizando o algoritmo Random Forest, de modo a realizar cálculo de predição de bioatividade em relação a alvos terapêuticos. 
    """
    )

    # Create tabs for modular workflow
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Upload de Dados", "Busca de Alvo", "Classificação", "Análise Gráfica", "Avaliação ADMET", "Geração de Modelos", "Predição"]
    )

    with tab1:
        st.header("Seleção de Dados")

        # File upload option
        uploaded_file = st.file_uploader(
            "Faça upload de um arquivo CSV com dados do alvo e classificações moleculares (opcional)",
            type=["csv"],
        )

        if uploaded_file is not None:
            st.success("Arquivo carregado com sucesso!")
            molecules_processed = pd.read_csv(uploaded_file)
            st.write("Preview dos dados carregados:")
            st.write(molecules_processed.head())

            if "compound_class" in molecules_processed.columns:
                st.success("Dados já classificados detectados.")
            else:
                st.warning(
                    "Os dados carregados não contêm classificações de compostos."
                )

            st.session_state["molecules_processed"] = molecules_processed

    with tab2:
        st.header("Busca de Alvo (Dados das plataformas ChEMBL e BindingDB)")

        col1, col2 = st.columns([3, 1])
        if "targets" not in st.session_state:
            st.session_state["targets"] = pd.DataFrame()
        with col1:
            search = st.text_input("Alvo")
        with col2:
            if st.button("Buscar"):
                with st.spinner("Buscando..."):
                    st.session_state["targets"] = search_target(search)

        targets = st.session_state["targets"]
        if not targets.empty:
            st.write(targets)
            st.write(f"Total de alvos encontrados: {len(targets)}")
            st.write(f"Alvos do ChEMBL: {len(targets[targets['source'] == 'ChEMBL'])}")
            st.write(
                f"Alvos do BindingDB: {len(targets[targets['source'] == 'BindingDB'])}"
            )

            selected_index = st.selectbox(
                "Selecione o alvo:",
                options=targets.index,
                format_func=lambda x: f"{targets.loc[x, 'pref_name']} ({targets.loc[x, 'source']})",
            )

            if selected_index is not None:
                with st.spinner("Selecionando base de dados de moléculas: "):
                    selected_molecules = select_target(selected_index, targets)

                if not selected_molecules.empty:
                    with st.spinner("Processando base de dados: "):
                        df_lipinski = lipinski(selected_molecules)
                        df_combined = pd.concat(
                            [selected_molecules, df_lipinski], axis=1
                        )
                        df_converted = convert_ugml_nm(df_combined)
                        df_labeled = label_bioactivity(df_converted)
                        df_norm = norm_value(df_labeled)
                        molecules_processed = pIC50(df_norm)
                        st.write(molecules_processed)
                        st.session_state["molecules_processed"] = molecules_processed

    with tab3:
        if "molecules_processed" in st.session_state:
            st.header("Classificação de Compostos")
            molecules_processed = st.session_state["molecules_processed"]

            if st.button("Realizar classificação de compostos"):
                classified_molecules = classify_compound(molecules_processed)
                if classified_molecules is not None:
                    st.write("Preview of classified molecules:")
                    st.write(classified_molecules.head())
                    st.session_state["molecules_processed"] = classified_molecules

                    class_summary = classified_molecules[
                        "compound_class"
                    ].value_counts()
                    st.subheader("Resumo de Classes de Compostos")
                    st.write(class_summary)

                    # Option to download the classified data
                    csv = classified_molecules.to_csv(index=False)
                    st.download_button(
                        label="Download classified data as CSV",
                        data=csv,
                        file_name="classified_molecules.csv",
                        mime="text/csv",
                    )

    with tab4:
        if "molecules_processed" in st.session_state:
            st.header("Análise Gráfica")
            molecules_processed = st.session_state["molecules_processed"]

            molecules_graph_analysis(molecules_processed)
            st.header("Teste de Mann-Whitney")
            df_mannwhitney = mannwhitney(molecules_processed)
            st.write(df_mannwhitney)

    with tab5:
        if "molecules_processed" in st.session_state:
            st.header("Avaliação ADMET")
            molecules_processed = st.session_state["molecules_processed"]

            rules_data = {
                "Regra": ["Lipinski", "Pfizer", "GSK", "Golden Triangle", "PAINS"],
                "Descrição": [
                    "Regra dos 5 de Lipinski",
                    "Regra de toxicidade da Pfizer",
                    "Regra da GSK",
                    "Regra do Triângulo Dourado",
                    "Filtro de Pan-Assay Interference Compounds",
                ],
                "Critérios": [
                    "MW ≤ 500; LogP ≤ 5; HBA ≤ 10; HBD ≤ 5",
                    "LogP > 3 e TPSA < 75",
                    "MW ≤ 400; LogP ≤ 4",
                    "200 ≤ MW ≤ 500; -2 ≤ LogP ≤ 5",
                    "Presença de subestruturas problemáticas",
                ],
            }
            rules_df = pd.DataFrame(rules_data)
            st.table(rules_df)

            smiles_list = molecules_processed["canonical_smiles"].tolist()
            with st.spinner("Realizando avaliação ADMET..."):
                admet_results = evaluate_admet(smiles_list)
                if not admet_results.empty:
                    st.write("Resultados ADMET detalhados:")
                    st.write(admet_results)

                    summary = summarize_results(admet_results)
                    st.write("Resumo da avaliação ADMET:")
                    for rule, result in summary.items():
                        st.write(f"{rule}: {result}")

    with tab6:
        if "molecules_processed" in st.session_state:
            st.header("Geração de Modelos")
            molecules_processed = st.session_state["molecules_processed"]

            model_col1, model_col2 = st.columns([0.5, 0.5])
            with model_col1:
                variance_input = st.number_input(
                    "Limite de variância:", min_value=0.0, value=0.1
                )
            with model_col2:
                estimators_input = st.number_input(
                    "Número de estimadores:", min_value=1, value=500
                )
            model_name = st.text_input("Nome para salvamento do modelo:")

            # Seleção de Algoritmos
            selected_algorithms = select_algorithms_ui()

            cleaned_data = detect_and_remove_outliers(molecules_processed)
            if st.button("Gerar modelos"):
                if model_name:
                    if selected_algorithms:
                        with st.spinner("Gerando modelos..."):
                            results, full_model_name = model_generation(
                                molecules_processed,
                                variance_input,
                                estimators_input,
                                model_name,
                                selected_algorithms,
                            )
                            if results is not None:
                                st.success(
                                    f"Modelos '{full_model_name}' gerados com sucesso!"
                                )
                                st.session_state["current_model_name"] = full_model_name
                    else:
                        st.error(
                            "Por favor, selecione pelo menos um algoritmo antes de gerar."
                        )
                else:
                    st.error(
                        "Por favor, forneça um nome para o modelo antes de gerá-lo."
                    )

            if st.button("Gerar modelos separados por classe"):
                if "compound_class" in molecules_processed.columns:
                    with st.spinner("Gerando modelos por classe"):
                        class_counts = generate_class_models(
                            molecules_processed,
                            variance_input,
                            estimators_input,
                            model_name,
                            selected_algorithms,
                        )
                        if class_counts:
                            st.success("Modelos gerados com sucesso!")
                            st.write("Resumo de moléculas usadas em cada classe:")
                            for class_name, count in class_counts.items():
                                st.write(f"- {class_name}: {count} moléculas")
                        else:
                            st.warning(
                                "Nenhum modelo foi gerado. Verifique se há moléculas suficientes em cada classe."
                            )
                else:
                    st.error(
                        "Por favor, execute a classificação dos compostos primeiro."
                    )

    # Sidebar for model prediction
    models = list_models()
    if models:
        with st.sidebar.header("1. Selecione o modelo a ser utilizado (alvo):"):
            selected_model_name = st.sidebar.selectbox("Modelo", models)
            selected_model = f"models/{selected_model_name}.pkl"

        with st.sidebar.header("2. Faça upload dos dados em CSV:"):
            uploaded_file = st.sidebar.file_uploader(
                "Faça upload do arquivo de entrada", type=["csv"]
            )
            st.sidebar.markdown(
                "[Exemplo de arquivo de entrada](https://raw.githubusercontent.com/ChiBeG/bioatividade_LIBV/main/exemplo.csv)"
            )

        if st.sidebar.button("Predizer"):
            with tab7:
                st.header("Cálculo de predição:")
                load_data = pd.read_csv(uploaded_file, sep=";", header=None, names=["SMILES", "ID", "Name"])
                load_data[["SMILES", "ID"]].to_csv("molecule.smi", sep=" ", header=False, index=False)

                st.header("**Dados originais de entrada**")
                st.write(load_data)

                with st.spinner("Calculando descritores..."):
                    desc_calc()

                st.header("**Cálculo de descritores moleculares realizado**")
                desc = pd.read_csv("descriptors_output.csv")
                st.write(desc)
                st.write(desc.shape)

                st.header("**Subconjunto de descritores do modelo selecionado**")
                try:
                    descriptor_list = pd.read_csv(f"descriptor_lists/{selected_model_name}_descriptor_list.csv")
                except FileNotFoundError:
                    st.warning(f"Arquivo de lista de descritores não encontrado para o modelo '{selected_model_name}'.")
                    st.info("Tentando usar todos os descritores disponíveis...")
                    descriptor_list = pd.DataFrame({"feature": desc.columns})

                Xlist = descriptor_list["feature"].tolist()
                desc_subset = desc[Xlist]
                st.write(desc_subset)
                st.write(desc_subset.shape)

                df_result = build_model(desc_subset, load_data, selected_model, selected_model_name)

                if not df_result.empty:
                    result_lipinski = lipinski(load_data)
                    df_final = pd.concat([df_result, result_lipinski], axis=1)
                    st.write(df_final)

                    st.header("Avaliação ADMET das Moléculas Preditas")
                    with st.spinner("Realizando avaliação ADMET..."):
                        smiles_list = load_data["SMILES"].tolist()
                        admet_results = evaluate_admet(smiles_list)

                        if not admet_results.empty:
                            st.write("Resultados ADMET detalhados:")
                            st.write(admet_results)

                            # Select only columns from admet_results that are not in df_final to avoid duplication
                            admet_columns_to_keep = [
                                col for col in admet_results.columns if col not in df_final.columns and col != "smiles"
                            ]

                            # Concatenate df_final with only non-duplicated columns from admet_results
                            df_final_with_admet = pd.concat([df_final, admet_results[admet_columns_to_keep]], axis=1)

                            st.header("Resultados Finais (Predição + ADMET)")
                            st.write(df_final_with_admet)



if __name__ == "__main__":
    main()
