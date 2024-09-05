import streamlit as st
import numpy as np
import pandas as pd
import subprocess
import os
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors


def desc_calc():
    # Performs the descriptor calculation
    try:
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove("molecule.smi")
    except Exception as e:
        st.error(f"Erro ao calcular descritores: {e}")


def lipinski(df, verbose=False):
    try:
        smiles = df["SMILES"] if "SMILES" in df.columns else df["canonical_smiles"]

        moldata = []
        for elem in smiles:
            mol = Chem.MolFromSmiles(elem)
            moldata.append(mol)

        baseData = np.arange(1, 1)
        i = 0
        for mol in moldata:
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

            row = np.array(
                [desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]
            )

            if i == 0:
                baseData = row
            else:
                baseData = np.vstack([baseData, row])
            i = i + 1

        columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
        descriptors = pd.DataFrame(data=baseData, columns=columnNames, index=df.index)

        return descriptors

    except Exception as e:
        st.error(f"Erro no cálculo dos descritores de Lipinski: {e}")
        return pd.DataFrame()
