import streamlit as st
import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client


def search_target(search):
    try:
        # ChEMBL search
        chembl_targets = search_chembl_target(search)

        # BindingDB search
        bindingdb_targets = search_bindingdb_target(search)

        # Combine results
        combined_targets = pd.concat(
            [chembl_targets, bindingdb_targets], ignore_index=True
        )

        return combined_targets
    except Exception as e:
        st.error(f"Erro na busca do alvo: {e}")
        return pd.DataFrame()


def search_chembl_target(search):
    target = new_client.target
    target_query = target.search(search)
    targets = pd.DataFrame.from_dict(target_query)
    targets["source"] = "ChEMBL"
    return targets


def search_bindingdb_target(search):
    base_url = "https://bindingdb.org/axis2/services/BDBService"
    endpoint = f"{base_url}/getLigandsByTarget"

    params = {
        "targetName": search,
        "cutoff": 100000,  # Set a high cutoff to get more results
        "response": "application/json",
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            # Process the data to create a DataFrame
            targets = pd.DataFrame(
                [
                    {
                        "target_chembl_id": f'BINDINGDB_{search.replace(" ", "_")}',
                        "pref_name": search,
                        "target_type": "SINGLE PROTEIN",
                        "organism": data[0].get("Target_Organism", "Unknown"),
                        "source": "BindingDB",
                    }
                ]
            )
            return targets

    return pd.DataFrame()


def select_target(selected_index, targets):
    try:
        selected_target = targets.loc[selected_index]

        if selected_target["source"] == "ChEMBL":
            return select_chembl_target(selected_target["target_chembl_id"])
        elif selected_target["source"] == "BindingDB":
            return select_bindingdb_target(selected_target["target_chembl_id"])
        else:
            st.error("Unknown source for the selected target")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Erro na seleção do alvo: {e}")
        return pd.DataFrame()


def select_chembl_target(target_chembl_id):
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_chembl_id).filter(
        standard_type="IC50"
    )
    df = pd.DataFrame.from_dict(res)

    st.header("Dados das moléculas")
    st.write(df)
    st.write(df.shape)

    df = units_filter(df)
    df = df[df.standard_value.notna()]
    df = df[df.canonical_smiles.notna()]
    df_clean = df.drop_duplicates(subset=["canonical_smiles"])

    selection = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "assay_description",
        "assay_type"
    ]
    df_selected = df_clean[selection]

    return df_selected


def select_bindingdb_target(target_id):
    base_url = "https://bindingdb.org/axis2/services/BDBService"
    endpoint = f"{base_url}/getLigandsByTarget"

    target_name = target_id.split("BINDINGDB_")[1].replace("_", " ")

    params = {
        "targetName": target_name,
        "cutoff": 100000,  # Set a high cutoff to get more results
        "response": "application/json",
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df["molecule_chembl_id"] = df["MonomerID"].apply(lambda x: f"BINDINGDB_{x}")
            df["canonical_smiles"] = df["Smiles"]
            df["standard_value"] = df["IC50_nM"]
            df["standard_units"] = "nM"

            st.header("Dados das moléculas")
            st.write(df)
            st.write(df.shape)

            df = units_filter(df)
            df = df[df.standard_value.notna()]
            df = df[df.canonical_smiles.notna()]
            df_clean = df.drop_duplicates(subset=["canonical_smiles"])

            selection = [
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_value",
                "standard_units",
            ]
            df_selected = df_clean[selection]

            return df_selected

    st.error("No data found for the selected BindingDB target")
    return pd.DataFrame()


def units_filter(df):
    units = ["nM", "ug.mL-1"]
    filtered_df = df.loc[df["standard_units"].isin(units)]
    return filtered_df
