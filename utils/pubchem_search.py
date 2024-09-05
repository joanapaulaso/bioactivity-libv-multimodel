import pandas as pd
import pubchempy as pcp
import streamlit as st
import requests


def search_pubchem_assays(query):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    search_url = f"{base_url}/assay/aids/JSON?{query}"
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("IdentifierList", {}).get("AID", [])
    else:
        st.error(f"Error searching PubChem: {response.status_code}")
        return []


def get_assay_data(aid):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    assay_url = f"{base_url}/assay/aid/{aid}/description/JSON"
    response = requests.get(assay_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching assay data: {response.status_code}")
        return None


def get_bioactivity_data(aid):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    bioactivity_url = f"{base_url}/assay/aid/{aid}/concise/JSON"
    response = requests.get(bioactivity_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching bioactivity data: {response.status_code}")
        return None
