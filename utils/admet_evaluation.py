import pandas as pd
from typing import List, Dict
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen


def evaluate_admet(smiles_list: List[str]) -> pd.DataFrame:
    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            properties = calculate_properties(mol)
            properties["smiles"] = smiles
            results.append(properties)
        else:
            st.warning(f"SMILES inválidos: {smiles}")

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("Não foram encontradas moléculas inválidas para avaliação.")
        return df

    # st.write("Available columns:")
    # st.write(df.columns.tolist())

    return evaluate_rules(df)


def calculate_properties(mol):
    return {
        "MW": Descriptors.ExactMolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "PAINS": check_pains(mol),
    }


def check_pains(mol):
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return 1 if catalog.HasMatch(mol) else 0


def evaluate_rules(df: pd.DataFrame) -> pd.DataFrame:
    # Lipinski Rule
    df["Lipinski_violations"] = (
        (df["MW"] > 500).astype(int)
        + (df["LogP"] > 5).astype(int)
        + (df["HBA"] > 10).astype(int)
        + (df["HBD"] > 5).astype(int)
    )
    df["Lipinski_result"] = df["Lipinski_violations"].apply(
        lambda x: "Excelente" if x < 2 else "Fraco"
    )

    # Pfizer Rule
    df["Pfizer_result"] = ((df["LogP"] > 3) & (df["TPSA"] < 75)).map(
        {True: "Fraco", False: "Excelente"}
    )

    # GSK Rule
    df["GSK_violations"] = (df["MW"] > 400).astype(int) + (df["LogP"] > 4).astype(int)
    df["GSK_result"] = df["GSK_violations"].apply(
        lambda x: "Excelente" if x == 0 else "Fraco"
    )

    # Golden Triangle
    df["GoldenTriangle_violations"] = (
        (df["MW"] < 200).astype(int)
        + (df["MW"] > 500).astype(int)
        + (df["LogP"] < -2).astype(int)
        + (df["LogP"] > 5).astype(int)
    )
    df["GoldenTriangle_result"] = df["GoldenTriangle_violations"].apply(
        lambda x: "Excelente" if x == 0 else "Fraco"
    )

    return df


def summarize_results(df: pd.DataFrame) -> Dict[str, str]:
    summary = {}

    for rule in ["Lipinski", "Pfizer", "GSK", "GoldenTriangle"]:
        if f"{rule}_result" in df.columns:
            Excelente_count = (df[f"{rule}_result"] == "Excelente").sum()
            total_count = len(df)
            if Excelente_count > total_count / 2:
                summary[rule] = "Aceito"
            else:
                summary[rule] = "Não Aceito"
        else:
            summary[rule] = "Não avaliado"

    # PAINS evaluation
    pains_count = df["PAINS"].sum()
    total_count = len(df)
    if pains_count < total_count / 2:
        summary["PAINS"] = "Aceito"
    else:
        summary["PAINS"] = "Não Aceito"

    return summary
