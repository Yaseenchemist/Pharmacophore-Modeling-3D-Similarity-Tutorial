"""
COMPLETE PHARMACOPHORE MODELING WITH ENHANCED 3D SIMILARITY VISUALIZATION

CONFIGURATION:
- Set your reference compound SMILES and name in the section below
- Place test compounds in 'smiles.smi' file
- Run the script to get comprehensive analysis with high-quality plots

FEATURES:
- 3D Similarity plots (PNG 600 DPI + SVG vector format)
- Individual pharmacophore analysis for each compound
- Distance matrices and feature distributions
- Detailed similarity reports
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Crippen
from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit import DataStructs
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import io
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION SECTION ====================
# ⚙️ SET YOUR REFERENCE COMPOUND HERE
# ضع معلومات المركب المرجعي هنا

REFERENCE_SMILES = "CNC(=O)C1=CC(OC2=CC=C(NC(=O)NC3=CC=C(Cl)C(=C3)C(F)(F)F)C=C2)=CC=N1"  # Example: "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
REFERENCE_NAME = "sorafinib"              # Example: "Ibuprofen"

# مثال للاستخدام - Example:
# REFERENCE_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
# REFERENCE_NAME = "Ibuprofen"

# أمثلة أخرى - More examples:
# Aspirin: REFERENCE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
# Paracetamol: REFERENCE_SMILES = "CC(=O)Nc1ccc(O)cc1"
# Sunitinib: REFERENCE_SMILES = "CCN(CC)CCNC(=O)C1=C(C)NC(/C=C2\C(=O)NC3=CC=C(F)C=C23)=C1C"
# ==============================================================

class CompletePharmacophoreAnalyzer:
    def __init__(self):
        """Initialize the complete system"""
        print("="*60)
        print("INITIALIZING PHARMACOPHORE ANALYSIS SYSTEM")
        print("="*60)
        
        # Load feature factory
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.feat_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        
        # Define color scheme for pharmacophore features
        self.feature_colors = {
            'Donor': '#2ECC40',        # Bright Green
            'Acceptor': '#FF4136',     # Bright Red
            'Aromatic': '#FFDC00',     # Gold
            'Hydrophobe': '#01C9E1',   # Cyan
            'PosIonizable': '#0074D9', # Blue
            'NegIonizable': '#B10DC9', # Purple
            'LumpedHydrophobe': '#FF851B' # Orange
        }
        
        # Feature symbols for labeling
        self.feature_symbols = {
            'Donor': 'D',
            'Acceptor': 'A',
            'Aromatic': 'R',
            'Hydrophobe': 'H',
            'PosIonizable': '+',
            'NegIonizable': '-',
            'LumpedHydrophobe': 'L'
        }
        
        print("✓ System initialized successfully!")
        print(f"✓ Available features: {', '.join(self.feature_colors.keys())}")
        print()
    
    def load_molecule(self, input_data, input_type='smiles', name=None):
        """Load molecule from SMILES or SDF"""
        print(f"Loading molecule ({input_type})...")
        if input_type.lower() == 'smiles':
            mol = Chem.MolFromSmiles(input_data)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {input_data}")
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        elif input_type.lower() == 'sdf':
            supplier = Chem.SDMolSupplier(input_data, removeHs=False)
            mol = next(supplier)
            if mol is None:
                raise ValueError(f"Could not read SDF file: {input_data}")
        else:
            raise ValueError("input_type must be 'smiles' or 'sdf'")
        
        if name:
            mol.SetProp("_Name", name)
        
        print(f"✓ Molecule loaded: {name if name else 'Unknown'}")
        print(f"  Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"  MW: {Descriptors.MolWt(mol):.2f}")
        print(f"  Atoms: {mol.GetNumAtoms()}")
        print(f"  Bonds: {mol.GetNumBonds()}")
        print()
        return mol
    
    def extract_pharmacophore_features(self, mol):
        """Extract pharmacophore features from molecule"""
        print("Extracting pharmacophore features...")
        print("-"*40)
        raw_features = self.feat_factory.GetFeaturesForMol(mol)
        features = []
        feature_summary = {}
        
        for feat in raw_features:
            feat_type = feat.GetFamily()
            atom_ids = feat.GetAtomIds()
            features.append((feat_type, atom_ids))
            
            if feat_type not in feature_summary:
                feature_summary[feat_type] = 0
            feature_summary[feat_type] += 1
        
        for feat_type, count in feature_summary.items():
            color_code = self.feature_colors.get(feat_type, '#888888')
            print(f"  {feat_type}: {count} features")
        
        print(f"\nTotal features found: {len(features)}")
        print()
        return features
    
    def calculate_pharmacophore_distances(self, mol, features):
        """Calculate distances between pharmacophore features"""
        print("Calculating inter-feature distances...")
        conf = mol.GetConformer()
        feature_coords = []
        feature_labels = []
        
        for i, (feat_type, atom_ids) in enumerate(features):
            coords = []
            for atom_id in atom_ids:
                pos = conf.GetAtomPosition(atom_id)
                coords.append([pos.x, pos.y, pos.z])
            
            centroid = np.mean(coords, axis=0)
            feature_coords.append(centroid)
            feature_labels.append(f"{self.feature_symbols.get(feat_type, 'X')}{i+1}")
        
        n_features = len(feature_coords)
        distances = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                dist = np.linalg.norm(
                    np.array(feature_coords[i]) - np.array(feature_coords[j])
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        df_distances = pd.DataFrame(
            distances,
            index=feature_labels,
            columns=feature_labels
        )
        
        print("-"*40)
        print("Key pharmacophore distances (Å):")
        count = 0
        for i in range(n_features):
            for j in range(i+1, n_features):
                if count < 5:
                    print(f"  {feature_labels[i]} <-> {feature_labels[j]}: {distances[i, j]:.2f} Å")
                    count += 1
        
        if count < (n_features * (n_features-1) / 2):
            print(f"  ... and {int(n_features * (n_features-1) / 2 - count)} more distances")
        print()
        
        return df_distances, feature_coords, feature_labels
    
    def generate_pharmacophore_fingerprint(self, mol):
        """Generate pharmacophore fingerprint"""
        print("Generating pharmacophore fingerprint...")
        sig_factory = Gobbi_Pharm2D.factory
        fp = Generate.Gen2DFingerprint(mol, sig_factory)
        on_bits = fp.GetOnBits()
        
        print(f"  Fingerprint size: {fp.GetNumBits()} bits")
        print(f"  Active bits: {len(on_bits)}")
        print(f"  Bit density: {len(on_bits)/fp.GetNumBits():.3f}")
        print()
        
        return fp
    
    def compare_molecules(self, mol1, mol2, name1="Molecule 1", name2="Molecule 2"):
        """Compare two molecules using pharmacophore fingerprints"""
        print(f"Comparing {name1} vs {name2}...")
        fp1 = self.generate_pharmacophore_fingerprint(mol1)
        fp2 = self.generate_pharmacophore_fingerprint(mol2)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        print(f"  Tanimoto similarity: {similarity:.3f}")
        
        if similarity > 0.8:
            print("  → Very similar pharmacophores")
        elif similarity > 0.6:
            print("  → Moderately similar pharmacophores")
        elif similarity > 0.4:
            print("  → Some similarity")
        else:
            print("  → Different pharmacophores")
        
        print()
        return similarity
    
    def visualize_2d_pharmacophore(self, mol, features, save_path=None):
        """Create 2D pharmacophore visualization"""
        print("Creating 2D pharmacophore visualization...")
        rdDepictor.Compute2DCoords(mol)
        
        highlight_atoms = []
        highlight_colors = {}
        
        for feat_type, atom_ids in features:
            color = self.feature_colors.get(feat_type, '#888888')
            rgb = tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))
            
            for atom_id in atom_ids:
                highlight_atoms.append(atom_id)
                highlight_colors[atom_id] = rgb
        
        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
        drawer.drawOptions().addAtomIndices = True
        drawer.drawOptions().addStereoAnnotation = True
        
        try:
            drawer.drawOptions().atomLabelFontSize = 16
        except AttributeError:
            try:
                drawer.drawOptions().annotationFontScale = 0.7
            except AttributeError:
                pass
        
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors
        )
        
        drawer.FinishDrawing()
        img_str = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_str))
        img = self._add_enhanced_legend(img, features)
        
        if save_path:
            img.save(save_path, dpi=(300, 300))
            print(f"  ✓ Saved: {save_path}")
        
        return img
    
    def _add_enhanced_legend(self, img, features):
        """Add enhanced legend to 2D image"""
        feature_counts = {}
        feature_atoms = {}
        
        for feat_type, atom_ids in features:
            if feat_type not in feature_counts:
                feature_counts[feat_type] = 0
                feature_atoms[feat_type] = []
            
            feature_counts[feat_type] += 1
            feature_atoms[feat_type].append(atom_ids)
        
        width, height = img.size
        legend_height = 150 + len(feature_counts) * 25
        new_height = height + legend_height
        new_img = Image.new('RGB', (width, new_height), 'white')
        new_img.paste(img, (0, 0))
        
        draw = ImageDraw.Draw(new_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_bold = font
        
        legend_y = height + 20
        legend_x = 30
        
        draw.text((legend_x, legend_y), "PHARMACOPHORE FEATURES", fill='black', font=font_bold)
        legend_y += 30
        
        for feat_type, color in self.feature_colors.items():
            if feat_type in feature_counts:
                draw.rectangle([legend_x, legend_y, legend_x+20, legend_y+20], 
                              fill=color, outline='black', width=2)
                
                text = f"{feat_type}: {feature_counts[feat_type]} features"
                draw.text((legend_x+30, legend_y+3), text, fill='black', font=font)
                
                if feature_atoms[feat_type]:
                    first_atoms = feature_atoms[feat_type][0]
                    atoms_str = f" (atoms: {', '.join(map(str, first_atoms[:3]))}"
                    if len(first_atoms) > 3:
                        atoms_str += "..."
                    atoms_str += ")"
                    draw.text((legend_x+200, legend_y+3), atoms_str, fill='gray', font=font)
                
                legend_y += 25
        
        legend_y += 10
        draw.text((legend_x, legend_y), 
                  "Note: Numbers on structure show atom indices", 
                  fill='gray', font=font)
        
        return new_img
    
    def visualize_3d_pharmacophore(self, mol, features, save_path=None):
        """Create 3D interactive pharmacophore visualization"""
        print("Creating 3D interactive pharmacophore...")
        conf = mol.GetConformer()
        fig = go.Figure()
        
        # Add bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_pos = conf.GetAtomPosition(begin_idx)
            end_pos = conf.GetAtomPosition(end_idx)
            
            fig.add_trace(go.Scatter3d(
                x=[begin_pos.x, end_pos.x],
                y=[begin_pos.y, end_pos.y],
                z=[begin_pos.z, end_pos.z],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add atoms
        atom_x, atom_y, atom_z = [], [], []
        atom_colors = []
        atom_labels = []
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            atom_x.append(pos.x)
            atom_y.append(pos.y)
            atom_z.append(pos.z)
            
            atom = mol.GetAtomWithIdx(i)
            element = atom.GetSymbol()
            atom_labels.append(f"{element}{i}")
            
            if element == 'C':
                atom_colors.append('black')
            elif element == 'N':
                atom_colors.append('blue')
            elif element == 'O':
                atom_colors.append('red')
            elif element == 'S':
                atom_colors.append('yellow')
            else:
                atom_colors.append('gray')
        
        fig.add_trace(go.Scatter3d(
            x=atom_x, y=atom_y, z=atom_z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=atom_colors,
                opacity=0.9,
                line=dict(color='darkgray', width=0.5)
            ),
            text=atom_labels,
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hovertemplate='<b>Atom %{text}</b><br>' +
                          'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                          '<extra></extra>',
            showlegend=False,
            name='Atoms'
        ))
        
        # Add pharmacophore features
        feature_traces = {}
        for feat_type, atom_ids in features:
            if feat_type not in feature_traces:
                feature_traces[feat_type] = {'x': [], 'y': [], 'z': []}
            
            coords = []
            for atom_id in atom_ids:
                pos = conf.GetAtomPosition(atom_id)
                coords.append([pos.x, pos.y, pos.z])
            
            centroid = np.mean(coords, axis=0)
            feature_traces[feat_type]['x'].append(centroid[0])
            feature_traces[feat_type]['y'].append(centroid[1])
            feature_traces[feat_type]['z'].append(centroid[2])
        
        for feat_type, positions in feature_traces.items():
            color = self.feature_colors.get(feat_type, '#888888')
            feature_labels = [f"{feat_type[:3]}{i+1}" for i in range(len(positions['x']))]
            
            fig.add_trace(go.Scatter3d(
                x=positions['x'],
                y=positions['y'],
                z=positions['z'],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color=color,
                    opacity=0.4,
                    line=dict(color=color, width=2)
                ),
                text=feature_labels,
                textposition='top center',
                textfont=dict(size=11, color=color, family='Arial Black'),
                name=feat_type,
                hovertemplate=f'<b>{feat_type}</b><br>' +
                              'Feature: %{text}<br>' +
                              'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                              '<extra></extra>'
            ))
        
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Molecule"
        
        fig.update_layout(
            title=f"3D Pharmacophore Model - {mol_name}",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.25),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            width=1280,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"  ✓ Saved: {save_path}")
        
        return fig
    
    def visualize_distance_heatmap(self, df_distances, save_path=None):
        """Create distance heatmap visualization"""
        print("Creating distance heatmap...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            df_distances,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Distance (Å)'},
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Pharmacophore Feature Distance Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
            plt.close()
        
        return fig
    
    def visualize_feature_distribution(self, features, mol_name="", save_path=None):
        """Create feature distribution visualization"""
        print("Creating feature distribution plots...")
        
        feature_counts = {}
        for feat_type, _ in features:
            if feat_type not in feature_counts:
                feature_counts[feat_type] = 0
            feature_counts[feat_type] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        features_list = list(feature_counts.keys())
        counts = list(feature_counts.values())
        colors = [self.feature_colors.get(f, '#888888') for f in features_list]
        
        # Bar chart
        bars = ax1.bar(features_list, counts, color=colors, edgecolor='black', linewidth=2)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Pharmacophore Feature Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        if feature_counts:
            wedges, texts, autotexts = ax2.pie(
                counts,
                labels=features_list,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(edgecolor='black', linewidth=2)
            )
            
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            
            ax2.set_title('Feature Proportions', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Pharmacophore Analysis - {mol_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
            plt.close()
        
        return fig
    
    def _calculate_descriptors(self, mol):
        """Calculate physicochemical descriptors for a molecule"""
        if mol is None:
            return {}
        
        Chem.SanitizeMol(mol)
        
        descriptors = {
            'LogP': Crippen.MolLogP(mol),
            'MW': Descriptors.MolWt(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'Rings': Descriptors.RingCount(mol),
            'AromaticRings': self._count_aromatic_rings(mol)
        }
        
        return descriptors
    
    def _count_aromatic_rings(self, mol):
        """Count aromatic rings in a molecule"""
        ring_info = Chem.GetSSSR(mol)
        aromatic_count = 0
        
        for ring in ring_info:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if all(atom.GetIsAromatic() for atom in atoms):
                aromatic_count += 1
        
        return aromatic_count
    
    def _get_similarity_category(self, similarity):
        """Categorize similarity score"""
        if similarity >= 80:
            return 'High Similarity (>80%)'
        elif similarity >= 60:
            return 'Moderate Similarity (60-80%)'
        elif similarity >= 40:
            return 'Low Similarity (40-60%)'
        else:
            return 'Very Low Similarity (<40%)'
    
    def create_3d_similarity_plot(self, molecules_data, reference_mol=None, reference_name="Reference", 
                                  save_path=None, plot_title="3D Scatter Plot of Molecular Similarity"):
        """Create an ENHANCED 3D scatter plot with high quality and clarity"""
        print("\n" + "="*60)
        print("CREATING ENHANCED 3D SIMILARITY PLOT")
        print("="*60)
        
        # Prepare data for plotting
        plot_data = []
        
        for mol, name, descriptors, similarity in molecules_data:
            plot_data.append({
                'Name': name,
                'LogP': descriptors.get('LogP', 0),
                'HBA': descriptors.get('HBA', 0),
                'HBD': descriptors.get('HBD', 0),
                'MW': descriptors.get('MW', 0),
                'TPSA': descriptors.get('TPSA', 0),
                'Similarity': similarity,
                'Category': self._get_similarity_category(similarity)
            })
        
        # Add reference compound if provided
        if reference_mol:
            ref_descriptors = self._calculate_descriptors(reference_mol)
            plot_data.append({
                'Name': reference_name,
                'LogP': ref_descriptors.get('LogP', 0),
                'HBA': ref_descriptors.get('HBA', 0),
                'HBD': ref_descriptors.get('HBD', 0),
                'MW': ref_descriptors.get('MW', 0),
                'TPSA': ref_descriptors.get('TPSA', 0),
                'Similarity': 100.0,
                'Category': 'Reference'
            })
        
        # Convert to DataFrame
        df_plot = pd.DataFrame(plot_data)
        
        # Create ENHANCED 3D plot with larger size and better quality
        fig = plt.figure(figsize=(24, 18))  # Much larger figure
        ax = fig.add_subplot(111, projection='3d')
        
        # Define color map
        color_map = {
            'High Similarity (>80%)': '#00FF00',
            'Moderate Similarity (60-80%)': '#FFFF00',
            'Low Similarity (40-60%)': '#FFA500',
            'Very Low Similarity (<40%)': '#FF0000',
            'Reference': '#0000FF'
        }
        
        # Plot each category
        for category in color_map.keys():
            cat_data = df_plot[df_plot['Category'] == category]
            if len(cat_data) > 0:
                scatter = ax.scatter(
                    cat_data['LogP'],
                    cat_data['HBA'],
                    cat_data['HBD'],
                    c=color_map[category],
                    s=cat_data['Similarity'] * 8 + 200,  # Larger markers
                    alpha=0.85,
                    edgecolors='black',
                    linewidths=2.5,
                    marker='*' if category == 'Reference' else 'o',
                    label=category,
                    depthshade=True
                )
        
        # Add ENHANCED labels
        for idx, row in df_plot.iterrows():
            label = f"{row['Name']}\n{row['Similarity']:.1f}%"
            z_offset = 0.4 if row['Category'] == 'Reference' else 0.25
            
            ax.text(
                row['LogP'],
                row['HBA'],
                row['HBD'] + z_offset,
                label,
                fontsize=13,  # Larger font
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='yellow' if row['Category'] == 'Reference' else 'white',
                    alpha=0.9,
                    edgecolor='black',
                    linewidth=2
                )
            )
        
        # ENHANCED plot customization
        ax.set_xlabel('Consensus Log P', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel('# H-bond Acceptors', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_zlabel('# H-bond Donors', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_title(plot_title, fontsize=22, fontweight='bold', pad=25)
        
        # Enhanced grid
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
        
        # CLOSER viewing angle for better clarity
        ax.view_init(elev=25, azim=40)
        
        # Adjust limits for closer view
        ax.set_xlim([df_plot['LogP'].min() - 0.5, df_plot['LogP'].max() + 0.5])
        ax.set_ylim([df_plot['HBA'].min() - 0.5, df_plot['HBA'].max() + 0.5])
        ax.set_zlim([df_plot['HBD'].min() - 0.5, df_plot['HBD'].max() + 0.5])
        
        # Enhanced legend
        ax.legend(loc='upper left', fontsize=13, frameon=True, 
                  fancybox=True, shadow=True, framealpha=0.95,
                  edgecolor='black', borderpad=1)
        
        # Larger tick labels
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        # Colorbar
        norm = plt.Normalize(df_plot['Similarity'].min(), df_plot['Similarity'].max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Similarity (%)', fontsize=15, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save in BOTH PNG (high DPI) and SVG formats
        if save_path:
            # Save as PNG with 600 DPI
            plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
            print(f"✓ 3D Similarity Plot (PNG 600 DPI) saved to: {save_path}")
            
            # Save as SVG (vector format)
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, dpi=600, bbox_inches='tight', format='svg')
            print(f"✓ 3D Similarity Plot (SVG vector) saved to: {svg_path}")
            
            plt.close()
        
        return fig, df_plot
    
    def _calculate_descriptor_similarity(self, desc1, desc2):
        """Calculate similarity based on descriptors"""
        key_descriptors = ['LogP', 'MW', 'HBA', 'HBD', 'TPSA']
        similarities = []
        
        for key in key_descriptors:
            if key in desc1 and key in desc2:
                val1, val2 = desc1[key], desc2[key]
                
                if key == 'LogP':
                    similarity = 1.0 / (1.0 + abs(val1 - val2))
                elif key == 'MW':
                    similarity = 1.0 / (1.0 + abs(val1 - val2) / 500)
                else:
                    max_val = max(abs(val1), abs(val2), 1)
                    similarity = 1.0 / (1.0 + abs(val1 - val2) / max_val)
                
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) * 100 if similarities else 0
        return avg_similarity
    
    def perform_comprehensive_similarity_analysis(self, reference_smiles, compounds_list, 
                                                  output_dir="similarity_output"):
        """Perform comprehensive similarity analysis"""
        print("\n" + "="*60)
        print("COMPREHENSIVE SIMILARITY ANALYSIS")
        print("="*60)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load reference
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
        
        reference_name = "Reference_Compound"
        reference_mol = Chem.AddHs(reference_mol)
        AllChem.EmbedMolecule(reference_mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(reference_mol)
        
        reference_descriptors = self._calculate_descriptors(reference_mol)
        print(f"\nReference Compound: {reference_name}")
        print(f"SMILES: {reference_smiles}")
        print("Descriptors:")
        for key, value in reference_descriptors.items():
            print(f"  {key}: {value:.2f}")
        
        # Analyze compounds
        results_data = []
        molecules_data = []
        
        print(f"\nAnalyzing {len(compounds_list)} compounds...")
        
        for smiles, name in compounds_list:
            print(f"\n  Processing: {name}")
            print(f"  SMILES: {smiles}")
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  ⚠ Warning: Invalid SMILES, skipping")
                continue
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            descriptors = self._calculate_descriptors(mol)
            
            fp1 = self.generate_pharmacophore_fingerprint(reference_mol)
            fp2 = self.generate_pharmacophore_fingerprint(mol)
            fp_similarity = DataStructs.TanimotoSimilarity(fp1, fp2) * 100
            
            descriptor_similarity = self._calculate_descriptor_similarity(
                reference_descriptors, descriptors
            )
            
            combined_similarity = (0.7 * fp_similarity + 0.3 * descriptor_similarity)
            
            print(f"  Pharmacophore Similarity: {fp_similarity:.1f}%")
            print(f"  Descriptor Similarity: {descriptor_similarity:.1f}%")
            print(f"  Combined Similarity: {combined_similarity:.1f}%")
            
            results_data.append({
                'Compound_Name': name,
                'SMILES': smiles,
                'LogP': descriptors['LogP'],
                'MW': descriptors['MW'],
                'HBA': descriptors['HBA'],
                'HBD': descriptors['HBD'],
                'TPSA': descriptors['TPSA'],
                'Rotatable_Bonds': descriptors['RotatableBonds'],
                'Rings': descriptors['Rings'],
                'Aromatic_Rings': descriptors['AromaticRings'],
                'Pharmacophore_Similarity': fp_similarity,
                'Descriptor_Similarity': descriptor_similarity,
                'Combined_Similarity': combined_similarity,
                'Similarity_Category': self._get_similarity_category(combined_similarity)
            })
            
            molecules_data.append((mol, name, descriptors, combined_similarity))
        
        df_results = pd.DataFrame(results_data)
        
        # Save CSV
        results_csv = os.path.join(output_dir, "similarity_results.csv")
        df_results.to_csv(results_csv, index=False)
        print(f"\n✓ Results saved to: {results_csv}")
        
        # Create enhanced 3D plot
        plot_path = os.path.join(output_dir, "3d_similarity_plot.png")
        fig, df_plot = self.create_3d_similarity_plot(
            molecules_data=molecules_data,
            reference_mol=reference_mol,
            reference_name=reference_name,
            save_path=plot_path,
            plot_title="3D Scatter Plot of Molecular Similarity to Reference Compound"
        )
        
        self._generate_similarity_report(df_results, reference_name, output_dir)
        
        return df_results, fig
    
    def _generate_similarity_report(self, df_results, reference_name, output_dir):
        """Generate detailed report"""
        report_path = os.path.join(output_dir, "similarity_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE SIMILARITY ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Reference Compound: {reference_name}\n")
            f.write(f"Total Test Compounds Analyzed: {len(df_results)}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("TOP 5 MOST SIMILAR COMPOUNDS:\n")
            f.write("-"*70 + "\n\n")
            
            top5 = df_results.nlargest(5, 'Combined_Similarity')
            for idx, row in top5.iterrows():
                f.write(f"{row['Compound_Name']}:\n")
                f.write(f"  Combined Similarity: {row['Combined_Similarity']:.1f}%\n")
                f.write(f"  Pharmacophore Similarity: {row['Pharmacophore_Similarity']:.1f}%\n")
                f.write(f"  Descriptor Similarity: {row['Descriptor_Similarity']:.1f}%\n")
                f.write(f"  LogP: {row['LogP']:.2f}, HBA: {row['HBA']}, HBD: {row['HBD']}\n")
                f.write(f"  MW: {row['MW']:.1f}, TPSA: {row['TPSA']:.1f}\n")
                f.write("-"*50 + "\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("SIMILARITY DISTRIBUTION:\n")
            f.write("-"*70 + "\n\n")
            
            categories = df_results['Similarity_Category'].value_counts()
            for category, count in categories.items():
                percentage = (count / len(df_results)) * 100
                f.write(f"{category}: {count} compounds ({percentage:.1f}%)\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("STATISTICAL SUMMARY:\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"Average Similarity: {df_results['Combined_Similarity'].mean():.1f}%\n")
            f.write(f"Maximum Similarity: {df_results['Combined_Similarity'].max():.1f}%\n")
            f.write(f"Minimum Similarity: {df_results['Combined_Similarity'].min():.1f}%\n")
            f.write(f"Standard Deviation: {df_results['Combined_Similarity'].std():.1f}%\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("OUTPUT FILES:\n")
            f.write("-"*70 + "\n\n")
            f.write("1. similarity_results.csv - Detailed results\n")
            f.write("2. 3d_similarity_plot.png - High-resolution PNG (600 DPI)\n")
            f.write("3. 3d_similarity_plot.svg - Vector graphics SVG\n")
            f.write("4. similarity_report.txt - This report\n")
        
        print(f"✓ Detailed report saved to: {report_path}")
    
    def complete_analysis(self, input_data, input_type='smiles', mol_name="Molecule", output_dir="output"):
        """Complete pharmacophore analysis for a single molecule"""
        print("\n" + "="*60)
        print(f"COMPLETE PHARMACOPHORE ANALYSIS: {mol_name}")
        print("="*60 + "\n")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}\n")
        
        mol = self.load_molecule(input_data, input_type, mol_name)
        features = self.extract_pharmacophore_features(mol)
        df_distances, feature_coords, feature_labels = self.calculate_pharmacophore_distances(mol, features)
        fingerprint = self.generate_pharmacophore_fingerprint(mol)
        
        print("="*40)
        print("GENERATING VISUALIZATIONS")
        print("="*40 + "\n")
        
        self.visualize_2d_pharmacophore(
            mol, features,
            save_path=os.path.join(output_dir, f"{mol_name}_2d_pharmacophore.png")
        )
        
        self.visualize_3d_pharmacophore(
            mol, features,
            save_path=os.path.join(output_dir, f"{mol_name}_3d_pharmacophore.html")
        )
        
        self.visualize_distance_heatmap(
            df_distances,
            save_path=os.path.join(output_dir, f"{mol_name}_distances.png")
        )
        
        self.visualize_feature_distribution(
            features, mol_name,
            save_path=os.path.join(output_dir, f"{mol_name}_distribution.png")
        )
        
        descriptors = self._calculate_descriptors(mol)
        self._save_analysis_results(mol, features, df_distances, fingerprint, descriptors, output_dir, mol_name)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        return mol, features, fingerprint, descriptors
    
    def _save_analysis_results(self, mol, features, df_distances, fingerprint, descriptors, output_dir, mol_name):
        """Save all analysis results to files"""
        features_data = []
        for i, (feat_type, atom_ids) in enumerate(features):
            features_data.append({
                'Feature_ID': f'{feat_type}_{i+1}',
                'Feature_Type': feat_type,
                'Atom_Indices': str(atom_ids),
                'Num_Atoms': len(atom_ids)
            })
        
        df_features = pd.DataFrame(features_data)
        df_features.to_csv(os.path.join(output_dir, f"{mol_name}_features.csv"), index=False)
        print(f"✓ Saved: {mol_name}_features.csv")
        
        df_distances.to_csv(os.path.join(output_dir, f"{mol_name}_distances.csv"))
        print(f"✓ Saved: {mol_name}_distances.csv")
        
        fp_bits = list(fingerprint.GetOnBits())
        fp_data = {
            'Fingerprint_Size': [fingerprint.GetNumBits()],
            'Active_Bits': [len(fp_bits)],
            'Bit_Density': [len(fp_bits)/fingerprint.GetNumBits()],
            'Active_Bit_Positions': [str(fp_bits[:100])]
        }
        
        df_fingerprint = pd.DataFrame(fp_data)
        df_fingerprint.to_csv(os.path.join(output_dir, f"{mol_name}_fingerprint_summary.csv"), index=False)
        print(f"✓ Saved: {mol_name}_fingerprint_summary.csv")
        
        mol_props = {
            'Property': ['Molecular_Formula', 'Molecular_Weight', 'Num_Atoms', 
                         'Num_Bonds', 'Num_Heavy_Atoms', 'Num_Rotatable_Bonds',
                         'Num_HBD', 'Num_HBA', 'TPSA', 'LogP'],
            'Value': [
                rdMolDescriptors.CalcMolFormula(mol),
                Descriptors.MolWt(mol),
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                descriptors['LogP']
            ]
        }
        
        df_props = pd.DataFrame(mol_props)
        df_props.to_csv(os.path.join(output_dir, f"{mol_name}_properties.csv"), index=False)
        print(f"✓ Saved: {mol_name}_properties.csv")
        
        df_descriptors = pd.DataFrame([descriptors])
        df_descriptors.to_csv(os.path.join(output_dir, f"{mol_name}_descriptors.csv"), index=False)
        print(f"✓ Saved: {mol_name}_descriptors.csv")


def main():
    """Main function"""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*15 + "ADVANCED PHARMACOPHORE MODELING" + " "*13 + "║")
    print("║" + " "*18 + "With Enhanced 3D Similarity Plot" + " "*12 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # Validate reference configuration
    if REFERENCE_SMILES == "ضع سمايل الرفرنس هنا":
        print("\n" + "!"*60)
        print("ERROR: Reference compound not configured!")
        print("!"*60)
        print("\nPlease edit the CONFIGURATION SECTION at the top:")
        print("1. Set REFERENCE_SMILES to your compound's SMILES")
        print("2. Set REFERENCE_NAME to your compound's name")
        print("\nExample:")
        print('REFERENCE_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"')
        print('REFERENCE_NAME = "Ibuprofen"')
        print("\n" + "!"*60)
        return
    
    # Validate SMILES
    test_mol = Chem.MolFromSmiles(REFERENCE_SMILES)
    if test_mol is None:
        print("\n" + "!"*60)
        print("ERROR: Invalid REFERENCE_SMILES!")
        print("!"*60)
        print(f"\nThe SMILES string is invalid:")
        print(f'REFERENCE_SMILES = "{REFERENCE_SMILES}"')
        print("\nPlease check and correct it.")
        print("!"*60)
        return
    
    analyzer = CompletePharmacophoreAnalyzer()
    
    smi_file = 'smiles.smi'
    if not os.path.exists(smi_file):
        print(f"Error: File {smi_file} not found!")
        return
    
    # Read compounds
    compounds = []
    with open(smi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 1:
                    smiles = parts[0]
                    name = parts[1] if len(parts) > 1 else f"Compound_{len(compounds)+1}"
                    compounds.append((smiles, name))
    
    print(f"Reference compound: {REFERENCE_NAME}")
    print(f"Found {len(compounds)} test compounds in {smi_file}\n")
    
    if len(compounds) == 0:
        print("No compounds found!")
        return
    
    # Similarity analysis
    print("="*60)
    print("PERFORMING COMPREHENSIVE SIMILARITY ANALYSIS")
    print("="*60)
    print(f"Using '{REFERENCE_NAME}' as reference")
    print(f"Reference SMILES: {REFERENCE_SMILES}\n")
    
    df_similarity, fig = analyzer.perform_comprehensive_similarity_analysis(
        reference_smiles=REFERENCE_SMILES,
        compounds_list=compounds,
        output_dir="similarity_analysis"
    )
    
    # Display results
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS RESULTS")
    print("="*60)
    
    df_sorted = df_similarity.sort_values('Combined_Similarity', ascending=False)
    print(f"\nREFERENCE: {REFERENCE_NAME} - 100.0% (Reference)")
    print("\nTest Compounds:")
    for idx, row in df_sorted.iterrows():
        print(f"{row['Compound_Name']}: {row['Combined_Similarity']:.1f}% similar")
    
    # Individual analysis
    print("\n" + "="*60)
    print("PERFORMING INDIVIDUAL COMPOUND ANALYSIS")
    print("="*60)
    
    analyzed_mols = []
    for smiles, name in compounds:
        print(f"\nAnalyzing {name}...")
        output_dir = f"output_{name}"
        
        try:
            mol, features, fp, descriptors = analyzer.complete_analysis(
                input_data=smiles,
                input_type='smiles',
                mol_name=name,
                output_dir=output_dir
            )
            analyzed_mols.append((mol, name, fp, descriptors))
            print(f"✓ Completed: {name}")
        except Exception as e:
            print(f"✗ Error: {name}: {str(e)}")
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*20 + "ANALYSIS COMPLETED SUCCESSFULLY!" + " "*19 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # Summary
    summary_file = "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PHARMACOPHORE ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Reference: {REFERENCE_NAME}\n")
        f.write(f"Reference SMILES: {REFERENCE_SMILES}\n")
        f.write(f"Total analyzed: {len(analyzed_mols)}\n")
        f.write("\nResults in: similarity_analysis/\n")
        f.write("Individual results in: output_*/ directories\n")
    
    print(f"✓ Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
