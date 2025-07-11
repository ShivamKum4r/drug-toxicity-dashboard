# ------------------- Imports -------------------
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import plotly.express as px
from rdkit.Chem import Draw
from torch_geometric.data import Batch
from rdkit.Chem import Descriptors


import time


# ------------------- Models -------------------
class ToxicityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

class RichGCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------- UI Setup -------------------
st.set_page_config(layout="wide", page_title="Drug Toxicity Predictor")
st.title("🧪 Drug Toxicity Prediction Dashboard")

# ------------------- Load Models with Spinner -------------------
# ------------------- Load Models with Temporary Messages -------------------
fp_model = ToxicityNet()
gcn_model = RichGCNModel()
fp_loaded = gcn_loaded = False

# Load Fingerprint Model
msg_fp = st.empty()
with msg_fp.container():
    with st.spinner("📦 Loading fingerprint model..."):
        time.sleep(6)
        try:
            fp_model.load_state_dict(torch.load("tox_model.pt", map_location=torch.device("cpu")))
            fp_model.eval()
            fp_loaded = True
            st.success("✅ Fingerprint model loaded.")
        except Exception as e:
            st.warning(f"⚠️ Fingerprint model not loaded: {e}")
    time.sleep(1)
    msg_fp.empty()

# Load GCN Model
msg_gcn = st.empty()
with msg_gcn.container():
    with st.spinner("📦 Loading GCN model..."):
        time.sleep(2)
        try:
            gcn_model.load_state_dict(torch.load("gcn_model.pt", map_location=torch.device("cpu")))
            gcn_model.eval()
            gcn_loaded = True
            st.success("✅ GCN model loaded.")
        except Exception as e:
            st.warning(f"⚠️ GCN model not loaded: {e}")
    time.sleep(1)
    msg_gcn.empty()

# Load Best Threshold
msg_threshold = st.empty()
with msg_threshold.container():
    with st.spinner("📊 Loading best threshold..."):
        time.sleep(2)
        try:
            best_threshold = float(np.load("gcn_best_threshold.npy"))
        except Exception as e:
            best_threshold = 0.5
            st.warning(f"⚠️ Using default threshold (0.5) for GCN model. Reason: {e}")
    st.success("✅ All models loaded. Dashboard is ready!")
    time.sleep(2)
    msg_threshold.empty()




# ------------------- Utility Functions -------------------
fp_gen = GetMorganGenerator(radius=2, fpSize=1024)

def get_molecule_info(mol):
    return {
        "Formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "Weight": round(Descriptors.MolWt(mol), 2),
        "Atoms": mol.GetNumAtoms(),
        "Bonds": mol.GetNumBonds()
    }



def predict_gcn(smiles):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None, None
    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        out = gcn_model(batch)
        prob = torch.sigmoid(out).item()
    return ("Toxic" if prob > best_threshold else "Non-toxic"), prob


def atom_feats(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumExplicitHs(),
        atom.GetNumImplicitHs(),
        atom.GetIsAromatic(),
        atom.GetMass(),
        int(atom.IsInRing()),
        int(atom.GetChiralTag()),
        int(atom.GetHybridization())
    ]

def smiles_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atoms = [atom_feats(a) for a in mol.GetAtoms()]
    if not atoms:
        return None  # No atoms present

    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    # Handle molecules with no bonds (e.g. single atom)
    if len(edges) == 0:
        edges = [[0, 0]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(atoms, dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, batch=batch)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


# def predict_gcn(smiles):
#     graph = smiles_to_graph(smiles)
#     if graph is None or graph.x.size(0) == 0:
#         return None, None
#     batch = Batch.from_data_list([graph])
#     with torch.no_grad():
#         out = gcn_model(batch)
#         raw = out.item()
#         prob = torch.sigmoid(out).item()
#     print(f"Raw logit: {raw:.4f}, Prob: {prob:.4f}")
#     return ("Toxic" if prob > best_threshold else "Non-toxic"), prob



# ------------------- Load Dataset -------------------
# df = pd.read_csv("tox21.csv")[['smiles', 'SR-HSE']].dropna()
# df = df[df['SR-HSE'].isin([0, 1])]

# # 🧼 Filter out invalid SMILES
# df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
# df = df[df['mol'].notna()].reset_index(drop=True)

df = pd.read_csv("tox21.csv")[['smiles', 'SR-HSE']].dropna()
df = df[df['SR-HSE'].isin([0, 1])].reset_index(drop=True)

# ✅ Filter invalid or unprocessable SMILES
def is_valid_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None and smiles_to_graph(smi) is not None

df = df[df['smiles'].apply(is_valid_graph)].reset_index(drop=True)




def create_graph_dataset(smiles_list, labels):
    data_list = []
    for smi, label in zip(smiles_list, labels):
        data = smiles_to_graph(smi, label)
        if data:
            data_list.append(data)
    return data_list

graph_data = create_graph_dataset(df['smiles'], df['SR-HSE'])
test_loader = DataLoader(graph_data, batch_size=32)

# ------------------- Plot Function -------------------
def plot_distribution(df, model_type, input_prob=None):
    col = 'fp_prob' if model_type == 'fp' else 'gcn_prob'
    df_plot = df[df[col].notna()].copy()
    df_plot["Label"] = df_plot["SR-HSE"].map({0: "Non-toxic", 1: "Toxic"})
    fig = px.histogram(df_plot, x=col, color="Label", nbins=30, barmode="overlay",
                       color_discrete_map={"Non-toxic": "green", "Toxic": "red"},
                       title=f"{model_type.upper()} Model - Test Set Distribution")
    if input_prob:
        fig.add_vline(x=input_prob, line_dash="dash", line_color="yellow", annotation_text="Your Input")
    return fig

# ------------------- Prediction Cache -------------------
@st.cache_data(show_spinner="Generating predictions...")

def predict_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES", 0.0
        fp = fp_gen.GetFingerprint(mol)
        fp_array = np.array(fp).reshape(1, -1)
        with torch.no_grad():
            logits = fp_model(torch.tensor(fp_array).float())
            prob = torch.sigmoid(logits).item()
        return ("Toxic" if prob > 0.5 else "Non-toxic"), prob
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def get_predictions(model_type='fp'):
    preds = []
    for smi in df['smiles']:
        try:
            p = predict_fp(smi)[1] if model_type == 'fp' else predict_gcn(smi)[1]
            preds.append(p)
        except:
            preds.append(None)
    return preds

df['fp_prob'] = get_predictions('fp') if fp_loaded else None
df['gcn_prob'] = get_predictions('gcn') if gcn_loaded else None

# ------------------- Evaluation Function -------------------
def evaluate_gcn_test_set(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to("cpu")  # Ensure on CPU
            out = model(batch)
            probs = torch.sigmoid(out)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    roc = roc_auc_score(all_labels, all_preds)

    df_eval = pd.DataFrame({
        "Predicted Probability": all_preds,
        "Label": ["Non-toxic" if i == 0 else "Toxic" for i in all_labels]
    })

    fig = px.histogram(df_eval, x="Predicted Probability", color="Label",
                       nbins=30, barmode="overlay",
                       color_discrete_map={"Non-toxic": "green", "Toxic": "red"},
                       title="GCN Test Set - Probability Distribution")
    fig.update_layout(bargap=0.1)

    st.success(f"✅ Accuracy: `{acc:.4f}`, ROC-AUC: `{roc:.4f}`")
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Tabs -------------------
tab1, tab2 = st.tabs(["🔬 Fingerprint Model", "🧬 GCN Model"])

with tab1:
    st.subheader("Fingerprint-based Prediction")
    with st.form("fp_form"):
        smiles_fp = st.text_input("Enter SMILES", "CCO")
        show_debug_fp = st.checkbox("🐞 Show Debug Info (raw score/logit)", key="fp_debug")
        predict_btn = st.form_submit_button("🔍 Predict")

    if predict_btn:
        with st.spinner("Predicting..."):
            mol = Chem.MolFromSmiles(smiles_fp)
            if mol:
                fp = fp_gen.GetFingerprint(mol)
                arr = np.array(fp).reshape(1, -1)
                tensor = torch.tensor(arr).float()
                with torch.no_grad():
                    output = fp_model(tensor)
                    prob = torch.sigmoid(output).item()
                    raw_score = output.item()
                    label = "Toxic" if prob > 0.5 else "Non-toxic"
                    color = "red" if label == "Toxic" else "green"

                st.markdown(f"<h4>🧾 Prediction: <span style='color:{color}'>{label}</span> — <code>{prob:.3f}</code></h4>", unsafe_allow_html=True)

                if show_debug_fp:
                    st.code(f"📉 Raw Logit: {raw_score:.4f}", language='text')
                    st.markdown("#### Fingerprint Vector (First 20 bits)")
                    st.code(str(arr[0][:20]) + " ...", language="text")

                st.image(Draw.MolToImage(mol), caption="Molecular Structure", width=250)

                info = get_molecule_info(mol)
                st.markdown("### Molecule Info:")
                for k, v in info.items():
                    st.markdown(f"**{k}:** {v}")

                st.plotly_chart(plot_distribution(df, 'fp', prob), use_container_width=True)
            else:
                st.error("❌ Invalid SMILES input. Please check your string.")

    with st.expander("📌 Example SMILES to Try"):
        st.markdown("""
        - `CCO` (Ethanol)  
        - `CC(=O)O` (Acetic Acid)  
        - `c1ccccc1` (Benzene)  
        - `CCN(CC)CC` (Triethylamine)  
        - `C1=CC=CN=C1` (Pyridine)
        """)

    with st.expander("🧪 Top 5 Toxic Predictions from Test Set (Fingerprint Model)"):
        if 'fp_prob' in df:
            top_toxic_fp = df[df['fp_prob'] > 0.5].sort_values('fp_prob', ascending=False)

            def is_valid_fp(smi):
                return Chem.MolFromSmiles(smi) is not None

            top_toxic_fp = top_toxic_fp[top_toxic_fp['smiles'].apply(is_valid_fp)].head(5)

            if not top_toxic_fp.empty:
                st.table(top_toxic_fp[['smiles', 'fp_prob']].rename(columns={'fp_prob': 'Predicted Probability'}))
            else:
                st.info("No valid top fingerprint predictions available.")
        else:
            st.info("Fingerprint model predictions not available.")


with tab2:
    st.subheader("Graph Neural Network Prediction")

    SUPPORTED_ATOMS = {1, 6, 7, 8, 9, 16, 17, 35, 53}  # H, C, N, O, F, S, Cl, Br, I

    def is_supported(mol):
        return all(atom.GetAtomicNum() in SUPPORTED_ATOMS for atom in mol.GetAtoms())

    with st.form("gcn_form"):
        smiles_gcn = st.text_input("Enter SMILES", "c1ccccc1", key="gcn_smiles")
        show_debug = st.checkbox("🐞 Show Debug Info (raw score/logit)")
        gcn_btn = st.form_submit_button("🔍 Predict")

    if gcn_btn:
        with st.spinner("Predicting..."):
            mol = Chem.MolFromSmiles(smiles_gcn)

            if mol is None:
                st.error("❌ Invalid SMILES: could not parse molecule.")
            elif not is_supported(mol):
                st.error("⚠️ This molecule contains unsupported atoms (e.g. Sn, P, etc.). GCN model only supports common organic elements.")
            else:
                graph = smiles_to_graph(smiles_gcn)
                if graph is None:
                    st.error("❌ SMILES is valid but could not be converted to graph. Possibly malformed structure.")
                else:
                    batch = Batch.from_data_list([graph])
                    with torch.no_grad():
                        out = gcn_model(batch)
                        prob = torch.sigmoid(out).item()
                        raw_score = out.item()
                        label = "Toxic" if prob > best_threshold else "Non-toxic"
                        color = "red" if label == "Toxic" else "green"

                    st.markdown(f"<h4>🧾 GCN Prediction: <span style='color:{color}'>{label}</span> — <code>{prob:.3f}</code></h4>", unsafe_allow_html=True)

                    if show_debug:
                        st.code(f"📉 Raw Logit: {raw_score:.4f}", language='text')

                    st.image(Draw.MolToImage(mol), caption="Molecular Structure", width=250)

                    def get_molecule_info(mol):
                        return {
                            "Molecular Weight": round(Chem.Descriptors.MolWt(mol), 2),
                            "LogP": round(Chem.Crippen.MolLogP(mol), 2),
                            "Num H-Bond Donors": Chem.Lipinski.NumHDonors(mol),
                            "Num H-Bond Acceptors": Chem.Lipinski.NumHAcceptors(mol),
                            "TPSA": round(Chem.rdMolDescriptors.CalcTPSA(mol), 2),
                            "Num Rotatable Bonds": Chem.Lipinski.NumRotatableBonds(mol)
                        }

                    info = get_molecule_info(mol)
                    st.markdown("### Molecule Info:")
                    for k, v in info.items():
                        st.markdown(f"**{k}:** {v}")

                    st.plotly_chart(plot_distribution(df, 'gcn', prob), use_container_width=True)

    with st.expander("📌 Example SMILES to Try"):
        st.markdown("""
        - `c1ccccc1` (Benzene)  
        - `C1=CC=CC=C1O` (Phenol)  
        - `CC(=O)OC1=CC=CC=C1C(=O)O` (Aspirin)  
        - `NCC(O)=O` (Glycine)  
        - `C1CCC(CC1)NC(=O)C2=CC=CC=C2` (Cyclohexylbenzamide)
        """)

    with st.expander("📥 Download GCN Model Predictions"):
        if 'gcn_prob' in df:
            def is_valid_gcn(smi):
                mol = Chem.MolFromSmiles(smi)
                return mol is not None and is_supported(mol) and smiles_to_graph(smi) is not None

            df_valid = df[df['smiles'].apply(is_valid_gcn)].copy()
            csv_gcn = df_valid[['smiles', 'gcn_prob', 'SR-HSE']].dropna().to_csv(index=False)
            st.download_button("Download CSV", csv_gcn, "gcn_predictions.csv", "text/csv")
        else:
            st.info("Predictions not available yet.")

    with st.expander("🧪 Top 5 Toxic Predictions from Test Set"):
        if 'gcn_prob' in df:
            def is_valid_gcn(smi):
                mol = Chem.MolFromSmiles(smi)
                return mol is not None and is_supported(mol) and smiles_to_graph(smi) is not None

            top_toxic = df[df['gcn_prob'] > best_threshold].copy()
            top_toxic = top_toxic[top_toxic['smiles'].apply(is_valid_gcn)]
            top_toxic = top_toxic.sort_values('gcn_prob', ascending=False).head(5)

            if not top_toxic.empty:
                st.table(top_toxic[['smiles', 'gcn_prob']].rename(columns={'gcn_prob': 'Predicted Probability'}))
            else:
                st.info("No valid top predictions available.")
        else:
            st.info("GCN model predictions not available.")

