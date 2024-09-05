from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

def smiles_to_image(smiles, size=(300, 300)):
    """Converte SMILES para uma imagem PNG usando RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=size)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            return img_byte_arr.getvalue()
        else:
            return None
    except Exception as e:
        print(f"Erro ao gerar imagem para SMILES {smiles}: {e}")
        return None


def get_molecular_image(smiles, size=(300, 300)):
    """Função wrapper para manter compatibilidade com o código existente."""
    return smiles_to_image(smiles, size)


def image_to_base64(img):
    if img is not None:
        buffered = BytesIO(img)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_str}" width="100">'
    return "Imagem não disponível"