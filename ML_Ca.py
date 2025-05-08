import os
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN, BrunnerNN_real
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_analyzer import OxideType, VoronoiAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
import pickle
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

def extract_features_from_cif(cif_path):
    """Extract comprehensive features from a CIF file for battery materials"""
    try:
        structure = Structure.from_file(cif_path)
        
        # 初始化分析器
        crystal_nn = CrystalNN()
        voronoi_nn = VoronoiNN()
        voronoi_analyzer = VoronoiAnalyzer()
        
        # 1. 基本晶体结构特性
        num_sites = len(structure)
        density = structure.density
        volume = structure.volume
        volume_per_atom = volume / num_sites
        
        # 晶格参数
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        # 2. 对称性特征
        try:
            spg_analyzer = SpacegroupAnalyzer(structure)
            spg_number = spg_analyzer.get_space_group_number()
            crystal_system = spg_analyzer.get_crystal_system()
            # 将晶系映射为数值特征
            crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 
                              'tetragonal', 'trigonal', 'hexagonal', 'cubic']
            crystal_system_id = crystal_systems.index(crystal_system) if crystal_system in crystal_systems else -1
        except:
            spg_number = 0
            crystal_system_id = -1
        
        # 3. 元素统计与电子结构特征
        elements = {}
        for site in structure:
            el = site.specie.symbol
            if el in elements:
                elements[el] += 1
            else:
                elements[el] = 1
        
        # 电池材料常见元素的分数
        common_elements = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al',  # 阳离子
                          'O', 'F', 'S', 'P', 'N',  # 阴离子
                          'Co', 'Ni', 'Mn', 'Fe', 'V', 'Ti', 'Cr',  # 过渡金属
                          'Si', 'Te', 'Se', 'Sb', 'Sn', 'Ge', 'C']  # 其他
        
        element_fractions = {}
        for el in common_elements:
            element_fractions[el] = elements.get(el, 0) / num_sites if num_sites > 0 else 0
        
        # 特别关注钙含量（修改为钙）
        ca_fraction = element_fractions.get('Ca', 0)
        
        # 阴离子/阳离子比例（影响离子导电性）
        anions = ['O', 'F', 'S', 'P', 'N']
        cations = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Co', 'Ni', 'Mn', 'Fe', 'V', 'Ti', 'Cr', 'Te']
        
        anion_count = sum([elements.get(el, 0) for el in anions])
        cation_count = sum([elements.get(el, 0) for el in cations])
        anion_cation_ratio = anion_count / cation_count if cation_count > 0 else 0
        
        # 4. 配位环境与多面体分析
        cn_data = []
        bond_lengths = []
        metal_oxygen_bonds = []
        
        # 计算每个原子的配位数和键长
        for i, site in enumerate(structure.sites):
            site_element = site.specie.symbol
            
            try:
                # 使用CrystalNN获取更准确的配位环境
                cn_info = crystal_nn.get_nn_info(structure, i)
                cn = len(cn_info)
                cn_data.append(cn)
                
                # 收集键长信息
                for neighbor in cn_info:
                    bond_length = neighbor['distance']
                    bond_lengths.append(bond_length)
                    
                    # 特别关注金属-氧键（对氧化物尤为重要）
                    neighbor_element = neighbor['site'].specie.symbol
                    if (site_element in cations and neighbor_element == 'O') or \
                       (site_element == 'O' and neighbor_element in cations):
                        metal_oxygen_bonds.append(bond_length)
            except:
                pass
                
        # 5. 计算多面体特征
        try:
            # Voronoi多面体分析
            voro_indices = []
            for i in range(len(structure)):
                try:
                    indices = voronoi_analyzer.analyze(structure, i).get_voronoi_polyhedra()
                    faces = len(indices)
                    voro_indices.append(faces)
                except:
                    voro_indices.append(0)
            
            avg_voro_faces = np.mean(voro_indices) if voro_indices else 0
            std_voro_faces = np.std(voro_indices) if voro_indices else 0
        except:
            avg_voro_faces = 0
            std_voro_faces = 0
                
        # 6. 计算统计特征
        # 配位数统计
        mean_cn = np.mean(cn_data) if cn_data else 0
        std_cn = np.std(cn_data) if cn_data else 0
        max_cn = np.max(cn_data) if cn_data else 0
        min_cn = np.min(cn_data) if cn_data else 0
        
        # 键长统计
        mean_bond_length = np.mean(bond_lengths) if bond_lengths else 0
        std_bond_length = np.std(bond_lengths) if bond_lengths else 0
        min_bond_length = np.min(bond_lengths) if bond_lengths else 0
        max_bond_length = np.max(bond_lengths) if bond_lengths else 0
        
        # 金属-氧键统计（对氧化物特别重要）
        mean_metal_o_bond = np.mean(metal_oxygen_bonds) if metal_oxygen_bonds else 0
        std_metal_o_bond = np.std(metal_oxygen_bonds) if metal_oxygen_bonds else 0
        
        # 7. 氧化物特有的特征
        oxide_type = -1  # 默认值
        try:
            if 'O' in elements:
                ot = OxideType(structure)
                if ot.is_peroxide:
                    oxide_type = 1
                elif ot.is_superoxide:
                    oxide_type = 2
                elif ot.is_ozonide:
                    oxide_type = 3
                else:
                    oxide_type = 0  # 普通氧化物
        except:
            pass
        
        # 8. 尝试估计氧化态（可能对电化学性能有影响）
        try:
            bva = BVAnalyzer()
            oxidation_structure = bva.get_valences(structure)
            # 计算平均氧化态
            oxidation_states = [float(site.specie.oxi_state) for site in oxidation_structure]
            mean_oxi = np.mean(oxidation_states)
            std_oxi = np.std(oxidation_states)
            max_oxi = np.max(oxidation_states)
            min_oxi = np.min(oxidation_states)
        except:
            mean_oxi = 0
            std_oxi = 0
            max_oxi = 0
            min_oxi = 0
            
        # 构建特征向量
        feature_vector = [
            # 基本结构特征
            num_sites,
            density,
            volume,
            volume_per_atom,
            a, b, c,
            alpha, beta, gamma,
            
            # 对称性特征
            spg_number,
            crystal_system_id,
            
            # 元素组成特征
            ca_fraction,  # 修改为钙含量
            anion_cation_ratio,
            
            # 配位环境
            mean_cn,
            std_cn,
            max_cn,
            min_cn,
            
            # 键长特征
            mean_bond_length,
            std_bond_length,
            min_bond_length,
            max_bond_length,
            
            # 金属-氧键特征
            mean_metal_o_bond,
            std_metal_o_bond,
            
            # Voronoi多面体特征
            avg_voro_faces,
            std_voro_faces,
            
            # 氧化物类型
            oxide_type,
            
            # 氧化态特征
            mean_oxi,
            std_oxi,
            max_oxi,
            min_oxi,
        ]
        
        # 添加常见元素分数
        for el in common_elements:
            feature_vector.append(element_fractions[el])
        
        return np.array(feature_vector)
        
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        # 返回合适长度的NaN数组，确保维度一致
        return np.full(32 + len(common_elements), np.nan)  # 32个基本特征 + 常见元素特征

def main(input_csv, cif_directory, output_csv):
    print("Loading dataset...")
    # 1. Load data
    data = pd.read_csv(input_csv, header=None, names=['id', 'capacity'])
    
    # 2. Split data into 8:1:1 ratio (train:validation:test)
    # First split: 90% train+val, 10% test
    train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    # Second split: 8/9 train, 1/9 validation (8:1 ratio within the 90%)
    train_data, val_data = train_test_split(train_val_data, test_size=1/9, random_state=42)
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # 3. Extract features for each set
    def extract_features_for_dataset(dataset, desc):
        features = []
        ids = dataset['id'].values
        y = dataset['capacity'].values
        
        for idx in tqdm(ids, desc=desc):
            cif_file_path = os.path.join(cif_directory, f'{idx}.cif')
            feature = extract_features_from_cif(cif_file_path)
            features.append(feature)
        
        # 检查是否有缺失值，并处理
        features_array = np.array(features)
        # 用列平均值填充NaN
        col_means = np.nanmean(features_array, axis=0)
        inds = np.where(np.isnan(features_array))
        features_array[inds] = np.take(col_means, inds[1])
        
        return features_array, ids, y
    
    print("Extracting features for training set...")
    X_train, train_ids, train_y = extract_features_for_dataset(train_data, "Training")
    
    print("Extracting features for validation set...")
    X_val, val_ids, val_y = extract_features_for_dataset(val_data, "Validation")
    
    print("Extracting features for test set...")
    X_test, test_ids, test_y = extract_features_for_dataset(test_data, "Testing")
    
    # 4. Train KRR model with validation set for hyperparameter tuning
    print("Tuning hyperparameters...")
    best_mae = float('inf')
    best_params = {}
    
    # Grid search parameters
    alphas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    gammas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    
    for alpha in tqdm(alphas, desc="Alpha"):
        for gamma in gammas:
            krr = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
            krr.fit(X_train, train_y)
            val_pred = krr.predict(X_val)
            mae = mean_absolute_error(val_y, val_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_params = {'alpha': alpha, 'gamma': gamma}
    
    print(f"Best parameters from validation: {best_params}")
    print(f"Best validation MAE: {best_mae}")
    
    # 5. Train final model with best parameters
    print("Training final model...")
    final_krr = KernelRidge(kernel='rbf', **best_params)
    
    # Combine training and validation sets for final training
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.concatenate((train_y, val_y))
    train_full_ids = np.concatenate((train_ids, val_ids))
    
    final_krr.fit(X_train_full, y_train_full)
    
    # 6. Make predictions
    train_pred = final_krr.predict(X_train_full)
    test_pred = final_krr.predict(X_test)
    
    # 7. Calculate MAE
    train_mae = mean_absolute_error(y_train_full, train_pred)
    test_mae = mean_absolute_error(test_y, test_pred)
    
    print(f"Training set MAE: {train_mae}")
    print(f"Test set MAE: {test_mae}")
    
    # 8. Save model
    with open('KRR_Ca_capacity_model.pkl', 'wb') as f:  # 改为Ca模型名称
        pickle.dump(final_krr, f)
    
    # 9. Create results CSV
    # Format: [train_id, train_target, train_pred, test_id, test_target, test_pred]
    results = []
    max_rows = max(len(train_full_ids), len(test_ids))
    
    for i in range(max_rows):
        row = []
        
        # 添加训练集数据
        if i < len(train_full_ids):
            row.extend([train_full_ids[i], y_train_full[i], train_pred[i]])
        else:
            row.extend(['', '', ''])
        
        # 添加测试集数据
        if i < len(test_ids):
            row.extend([test_ids[i], test_y[i], test_pred[i]])
        else:
            row.extend(['', '', ''])
        
        results.append(row)
    
    # 保存为CSV（不带表头）
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False, header=False)
    
    print(f"Results saved to {output_csv}")
    print(f"Final Test MAE: {test_mae}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KRR model for Ca-ion battery material prediction")
    parser.add_argument("--input_csv", type=str, default="/home/user1/code/data/material_Ca/id_prop.csv",
                        help="Input CSV file with IDs and target values")
    parser.add_argument("--cif_dir", type=str, default="/home/user1/code/data/material_Ca/",
                        help="Directory containing CIF files")
    parser.add_argument("--output_csv", type=str, default="ML_results.csv",
                        help="Output CSV file for results")
    
    args = parser.parse_args()
    main(args.input_csv, args.cif_dir, args.output_csv)
