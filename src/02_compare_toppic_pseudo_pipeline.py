# Copyright (c) 2014 - 2024, The Trustees of Indiana University.
##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
# http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/python3

import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from statistics import mean, stdev
import os
import json
import math
import statistics
import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from feature import Feature
from proteoform import Proteoform
from itertools import groupby

AXEX_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_mz(mass, charge):
    return (mass + (charge * 1.00727)) / charge


def get_features(feature_file):
    df = pd.read_csv(feature_file)
    features = []
    for i in range(0, len(df)):
        data = df.iloc[i]
        if 'Score' in data:
            score = data['Score']
            if math.isnan(score):
                score = 0
        else:
            score = data['ecscore']
            if math.isnan(score):
                score = 0

        if 'XIC' in df.keys():
            split_xic = data['XIC'].split(';')
            xic = [float(v) for v in split_xic]
        else:
            xic = []

        if 'Envelope' in df.keys():
            split_Envelope = data['Envelope'].split(';')
            envelope = [(float(peak.split('&')[0]), float(peak.split('&')[1])) for peak in split_Envelope]
        else:
            envelope = []

        if 'Envelope' in df.keys():
            rt_apex = data['rtApex']
        else:
            rt_apex = 0

        features.append(Feature(data['ID'], data['Mass'], data['MonoMz'], data['Charge'], data['Intensity'],
                        data['mzLo'], data['mzHi'], data['rtLo'], data['rtHi'], rt_apex, score, xic, envelope))
    return features


def return_fragments(prsm_file):
    dataFile = open(prsm_file)
    data = dataFile.read()
    jsonObj = json.loads(data[data.find('{'): data.rfind('}')+1])
    dataFile.close()
    peaks = jsonObj['prsm']['ms']['peaks']['peak']
    matched_peak_list = []
    for peak in peaks:
        if len(peak) < 8:
            continue
        peak_info = (float(peak['monoisotopic_mass']), int(peak['charge']), float(peak['intensity']), float(peak['monoisotopic_mz']))
        matched_peak_list.append(peak_info)
    return matched_peak_list


def get_proteoform(proteoform_file, prsm_dir, inte_dict=None, dtype="DDA"):
    df = pd.read_csv(proteoform_file, skiprows=35, delimiter="\t")
    proteoforms = []
    for i in range(0, len(df)):
        data = df.iloc[i]

        spectrum_id = -1
        if 'Spectrum ID' in data:
            spectrum_id = data['Spectrum ID']

        prsm_id = -1
        if 'Prsm ID' in data:
            prsm_id = data['Prsm ID']

        rt = -1
        if 'Retention time' in data:
            if dtype == "DDA":
                rt = data['Retention time']/60
            else:
                rt = data['Retention time']

        apex_rt = -1
        if 'Feature apex time' in data:
            apex_rt = data['Feature apex time']

        charge = -1
        if 'Charge' in data:
            charge = data['Charge']

        mass = -1
        if 'Precursor mass' in data:
            mass = data['Precursor mass']

        adjusted_mass = -1
        if 'Precursor mass' in data:
            adjusted_mass = data['Adjusted precursor mass']

        seq = ""
        if 'Proteoform' in data:
            seq = data['Proteoform'][2:-2]

        peak = -1
        if '#peaks' in data:
            peak = data['#peaks']

        matched_fragment = -1
        if '#matched fragment ions' in data:
            matched_fragment = data['#matched fragment ions']

        feature_intensity = -1
        if 'Feature intensity' in data:
            if data['Feature intensity'] != '-':
                feature_intensity = float(data['Feature intensity'])
            elif spectrum_id >= 0:
                feature_intensity = inte_dict[spectrum_id]

        accession = ""
        if 'Protein accession' in data:
            accession = data['Protein accession']

        score = -1
        if 'Feature score' in data:
            score = data['Feature score']

        evalue = -1
        if 'E-value' in data:
            evalue = data['E-value']

        first_residue = -1
        if 'First residue' in data:
            first_residue = data['First residue']

        last_residue = -1
        if 'Last residue' in data:
            last_residue = data['Last residue']

        acetylation = False
        proteoform = Proteoform(i, spectrum_id, prsm_id, rt, apex_rt, charge, mass, seq, acetylation,
                                score, peak, matched_fragment, feature_intensity, accession, evalue)
        proteoform.first_residue = first_residue
        proteoform.last_residue = last_residue
        proteoform.adjusted_mass = adjusted_mass
        proteoform.status = False

        if prsm_dir is not None:
            prsm_file = os.path.join(prsm_dir, "prsm" + str(prsm_id) + ".js")
            matched_peaks = return_fragments(prsm_file)
            proteoform.matched_peaks = matched_peaks
        proteoforms.append(proteoform)
    return proteoforms


def _binary_search(DDA_proteoforms_copy, prec_mass):
    low = 0
    mid = 0
    high = len(DDA_proteoforms_copy) - 1
    while low <= high:
        mid = (high + low) // 2
        if DDA_proteoforms_copy[mid].mass < prec_mass:
            low = mid + 1
        elif DDA_proteoforms_copy[mid].mass > prec_mass:
            high = mid - 1
    return low


def _getExtendMasses(mass):
    IM = 1.00235
    extend_offsets_ = [0, -IM, IM, 2 * -IM, 2 * IM, 3 * -IM, 3 * IM]
    result = []
    for i in range(0, len(extend_offsets_)):
        new_mass = mass + extend_offsets_[i]
        result.append(new_mass)
    return result


def _getMatchedProteoforms(DDA_proteoforms_copy, proteoform, tolerance):
    prec_mass = proteoform.mass
    error_tole = prec_mass * tolerance
    ext_masses = _getExtendMasses(prec_mass)
    min_idx = _binary_search(DDA_proteoforms_copy, ext_masses[len(ext_masses) - 2] - (2 * error_tole))
    max_idx = _binary_search(DDA_proteoforms_copy, ext_masses[len(ext_masses) - 1] + (2 * error_tole))
    temp_proteoforms = [DDA_proteoforms_copy[f_idx] for f_idx in range(min_idx, max_idx) if not hasattr(DDA_proteoforms_copy[f_idx], 'used')]
    matched_proteoforms = []
    for temp_proteoform in temp_proteoforms:
        for k in range(0, len(ext_masses)):
            mass_diff = abs(ext_masses[k] - temp_proteoform.mass)
            if (mass_diff <= error_tole):
                matched_proteoforms.append(temp_proteoform)
                break
    return matched_proteoforms


def _getOverlappingProteoforms(temp_proteoforms, proteoform, time_tol):
    overlapping = []
    for p_idx in range(0, len(temp_proteoforms)):
        p = temp_proteoforms[p_idx]
        if abs(p.rt - proteoform.rt) < 5:
            overlapping.append(p)
    return overlapping


def get_common_proteoforms(DIA_proteoforms, DDA_proteoforms, tolerance=15E-6, time_tol=5):
    DIA_proteoforms_copy = deepcopy(DIA_proteoforms)
    DIA_proteoforms_copy.sort(key=lambda x: (x is None, x.intensity), reverse=True)
    DDA_proteoforms_copy = deepcopy(DDA_proteoforms)
    DDA_proteoforms_copy.sort(key=lambda x: (x is None, x.mass), reverse=False)
    # Get common features based on the RT overlap
    common_proteoforms = []
    for idx in range(0, len(DIA_proteoforms_copy)):
        proteoform = DIA_proteoforms_copy[idx]
        if hasattr(proteoform, 'used'):
            continue
        else:
            overlapping_proteoforms = _getMatchedProteoforms(DDA_proteoforms_copy, proteoform, tolerance)
            if len(overlapping_proteoforms) > 0:
                apex_diff = [abs(proteoform.rt - p.rt) for p in overlapping_proteoforms]
                selected_proteoform = overlapping_proteoforms[apex_diff.index(min(apex_diff))]
                DDA_proteoforms_copy[DDA_proteoforms_copy.index(selected_proteoform)].used = True
                common_proteoforms.append((proteoform, selected_proteoform))
    return common_proteoforms


def assign_rt(proteoforms, ms1_features):
    if ms1_features is None:
        for p in proteoforms:
            p.rt_low = 0
            p.rt_high = 0
            p.mz = get_mz(p.mass, p.charge)
            p.envelope = []
            p.xic = []
            p.score = 0
            p.length = 0
    else:
        t = []
        for p_idx in range(len(proteoforms)):
            p = proteoforms[p_idx]
            for mf in ms1_features:
                ppm_mass_diff = (abs(mf.mass-p.mass)/mf.mass)*1e6
                # if abs(round(mf.mass, 3) - round(p.mass, 3)) < 0.01 and mf.charge == p.charge:
                if ppm_mass_diff < 5 and mf.charge == p.charge:
                    if p.rt_apex == '-' or mf.rt_low <= p.rt_apex/60 <= mf.rt_high:
                        t.append(p_idx)
                        p.rt_low = mf.rt_low
                        p.rt_high = mf.rt_high
                        p.mz = get_mz(p.mass, p.charge)
                        p.envelope = mf.envelope
                        p.xic = mf.xic
                        p.score = mf.ecscore
                        p.length = np.count_nonzero(mf.xic)
                        break


def get_isolation_window_ms1_features_maxInte(DIA_proteoforms_temp, base_mz, windowSize):
    isolation_window_start = base_mz - windowSize/2
    isolation_window_end = base_mz + windowSize/2
    win_proteoforms = []
    for proteoforms_idx in range(0, len(DIA_proteoforms_temp)):
        if hasattr(DIA_proteoforms_temp[proteoforms_idx], 'envelope'):
            total_inte = 0
            assigned_inte = 0
            for peak in DIA_proteoforms_temp[proteoforms_idx].envelope:
                total_inte = total_inte + peak[1]
                if isolation_window_start <= peak[0] < isolation_window_end:
                    assigned_inte = assigned_inte + peak[1]
            coverage = assigned_inte/total_inte
            if coverage >= 0.5:
                win_proteoforms.append(DIA_proteoforms_temp[proteoforms_idx])
    return win_proteoforms


def get_overlap(f, feature):
    start_rt = max(feature.rt_low, f.rt_low)
    end_rt = min(feature.rt_high, f.rt_high)
    return (start_rt, end_rt)


def get_common_peaks(matched_peaks_1, matched_peaks_2):
    matched = []
    for p1 in matched_peaks_1:
        for p2 in matched_peaks_2:
            if abs(round(float(p1[0]), 3) - round(float(p2[0]), 3)) < 0.001:
                matched.append((p1[0], p2[0]))
                break
    return len(matched)/((len(matched_peaks_1)+len(matched_peaks_1))/2)


def explore_isolation_windows(DIA_proteoforms_temp, DIA_ms1_features, isolation_window_base_selected, windowSize):
    assign_rt(DIA_proteoforms_temp, DIA_ms1_features)
    prots = []
    for base_mz in isolation_window_base_selected:
        win_proteoforms = get_isolation_window_ms1_features_maxInte(DIA_proteoforms_temp, base_mz, windowSize)
        win_proteoforms.sort(key=lambda x: (x is None, x.intensity), reverse=False)
        for i in range(0, len(win_proteoforms)):
            selected_proteoform1 = win_proteoforms[i]
            if hasattr(selected_proteoform1, 'used') and selected_proteoform1.used is True:
                continue
            pairs = []
            for j in range(i+1, len(win_proteoforms)):
                selected_proteoform2 = win_proteoforms[j]
                if hasattr(selected_proteoform2, 'used') and selected_proteoform2.used is True:
                    continue
                start_rt, end_rt = get_overlap(selected_proteoform1, selected_proteoform2)
                overlapping_rt_range = (end_rt - start_rt) >= 0
                if overlapping_rt_range:
                    if selected_proteoform1.accession == selected_proteoform2.accession:
                        matched_peaks_1 = selected_proteoform1.matched_peaks
                        matched_peaks_2 = selected_proteoform2.matched_peaks
                        shared_peaks_percent = get_common_peaks(matched_peaks_1, matched_peaks_2)
                        if shared_peaks_percent > 0.5:
                            pairs.append((i, j))
            if len(pairs) == 0:
                prots.append(selected_proteoform1)
            else:
                for pair in pairs:
                    idx_i = pair[0]
                    idx_j = pair[1]
                    selected_proteoform1 = win_proteoforms[idx_i]
                    selected_proteoform2 = win_proteoforms[idx_j]

                    if len(selected_proteoform1.matched_peaks) > len(selected_proteoform2.matched_peaks):
                        if selected_proteoform1 not in prots:
                            win_proteoforms[idx_i].used = True
                            prots.append(selected_proteoform1)
                    else:
                        if selected_proteoform2 not in prots:
                            win_proteoforms[idx_j].used = True
                            prots.append(selected_proteoform2)

    for p in prots:
        if hasattr(p, 'used'):
            delattr(p, "used")
    for p in DIA_proteoforms_temp:
        if not hasattr(p, 'envelope'):
            prots.append(p)
    prots.sort(key=lambda x: (x is None, x.i), reverse=False)
    return prots


def get_matched_peaks(common_proteoforms):
    DIA_matched_peaks = []
    DDA_matched_peaks = []
    DIA_num_peaks = []
    DDA_num_peaks = []
    for pair in common_proteoforms:
        DIA_matched_peaks.append(pair[0].matched_peaks)
        DDA_matched_peaks.append(pair[1].matched_peaks)
        DIA_num_peaks.append(pair[0].num_peaks)
        DDA_num_peaks.append(pair[1].num_peaks)
    return DIA_matched_peaks, DDA_matched_peaks, DIA_num_peaks, DDA_num_peaks


def __get_end_index(all_lines, begin_idx):
    idx = begin_idx
    while (idx < len(all_lines) and "END IONS" not in all_lines[idx]):
        idx = idx + 1
    return idx


def get_proteoform_inte_dict(filename):
    file = open(filename)
    all_lines = file.readlines()
    all_lines = [x.strip() for x in all_lines]
    file.close()
    inte_dict = {}

    begin_idx = 0
    while (begin_idx < len(all_lines)):
        end_idx = __get_end_index(all_lines, begin_idx)
        spec_lines = all_lines[begin_idx:end_idx + 1]
        begin_idx = end_idx + 1
        if begin_idx >= len(all_lines):
            break
        for line in spec_lines:
            if 'ID=' == line[0:3] or 'SPECTRUM_ID=' in line:
                spec_id = int(line.split('=')[1])
            if 'PRECURSOR_INTENSITY=' in line:
                if 'PRECURSOR_INTENSITY=' == line:
                    intensity = 0
                else:
                    if ':' in line:
                        intensity = float(line.split('=')[1].split(':')[0])
                    else:
                        intensity = float(line.split('=')[1])
        inte_dict[spec_id] = intensity
    return inte_dict


def remove_duplicate_proteoforms(proteoforms_original, ppm_tole=10E-6):
    proteoforms_bk = deepcopy(proteoforms_original)
    for p in proteoforms_bk:
        p.status = False
    IM = 1.00235

    # cluster proteoforms by protein accession
    proteoforms_bk.sort(key=lambda x: x.accession)
    grouped_proteoforms = {}
    for accession, group in groupby(proteoforms_bk, key=lambda x: x.accession):
        grouped_proteoforms[accession] = list(group)
    new_proteoforms = []
    for key, objects in grouped_proteoforms.items():
        if len(objects) > 1:
            objects.sort(key=lambda x: x.mass)
            for i in range(len(objects)):
                obj1 = objects[i]
                for j in range(i + 1, len(objects)):
                    obj2 = objects[j]
                    mass_tole = ppm_tole * max(obj1.mass, obj2.mass)
                    if (abs(obj1.mass - obj2.mass) <= mass_tole or abs(obj1.mass - obj2.mass - IM) <= mass_tole or abs(obj1.mass - obj2.mass + IM) <= mass_tole):
                        if obj1.intensity >= obj2.intensity:
                            if obj1.status == False and obj2.status == False:
                                new_proteoforms.append(obj1)
                        else:
                            if obj1.status == False and obj2.status == False:
                                new_proteoforms.append(obj2)
                        obj1.status = True
                        obj2.status = True
            for i in range(len(objects)):
                obj = objects[i]
                if obj.status == False:
                    new_proteoforms.append(obj)
        else:
            new_proteoforms.append(objects[0])
    return new_proteoforms


if __name__ == '__main__':
    # file_dir = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "TopDIA_Published_Data"), r"04_test_data")
    mass_ranges = ["720_800", "800_880", "880_960", "960_1040", "1040_1120", "1120_1200"]
    isolation_window_base = [[722, 726, 730, 734, 738, 742, 746, 750, 754, 758, 762, 766, 770, 774, 778, 782, 786, 790, 794, 798],
                             [802, 806, 810, 814, 818, 822, 826, 830, 834, 838, 842, 846, 850, 854, 858, 862, 866, 870, 874, 878],
                             [882, 886, 890, 894, 898, 902, 906, 910, 914, 918, 922, 926, 930, 934, 938, 942, 946, 950, 954, 958],
                             [962, 966, 970, 974, 978, 982, 986, 990, 994, 998, 1002, 1006, 1010, 1014, 1018, 1022, 1026, 1030, 1034, 1038],
                             [1042, 1046, 1050, 1054, 1058, 1062, 1066, 1070, 1074, 1078, 1082, 1086, 1090, 1094, 1098, 1102, 1106, 1110, 1114, 1118],
                             [1122, 1126, 1130, 1134, 1138, 1142, 1146, 1150, 1154, 1158, 1162, 1166, 1170, 1174, 1178, 1182, 1186, 1190, 1194, 1198]]

    all_dda_proteins_data = []
    all_dia_proteins_data = []
    all_common_proteins_data = []
    all_dda_proteoforms_data = []
    all_dia_proteoforms_data = []
    all_common_proteoforms_data = []

    dda_unique_proteoform_data = []
    dia_unique_proteoform_data = []
    common_proteoform_data = []
    dda_unique_protein_data = []
    dia_unique_protein_data = []
    common_protein_data = []

    file_dir = r"C:\Users\Abdul\Tulane University\Liu, Kevin - 202210_topdia_abdul\TopDIA_Published_Data\04_test_data"
    replicates = [2, 3]
    for replicate in replicates:
        print("Processing Replicate:", replicate)
        DDA_proteoforms_data = []
        DIA_proteoforms_data = []
        all_dda_proteins = []
        all_dia_proteins = []
        all_common_proteins = []
        all_dda_proteoforms = []
        all_dia_proteoforms = []
        all_common_proteoforms = []
        DIA_matched_peaks_shared = []
        DDA_matched_peaks_shared = []
        DIA_num_peaks_shared = []
        DDA_num_peaks_shared = []
        DIA_inte_common = []
        DDA_inte_common = []
        DIA_inte_unique = []
        DDA_inte_unique = []
        for mass_range_idx in range(0, len(mass_ranges)):
            mass_range = mass_ranges[mass_range_idx]
            target_file = "20231117_DIA_" + mass_range + "_rep" + str(replicate)
            proteoform_file = os.path.join(file_dir, target_file + '_ms2_toppic_proteoform_single.tsv')
            prsm_dir = os.path.join(file_dir, target_file + "_html\prsms")
            ms1_feature_file = os.path.join(os.path.join(file_dir, "ms1_features_mzrt_toppic_pipeline"), target_file + "_frac.mzrt.csv")
            ms1_features = get_features(ms1_feature_file)
            proteoforms_original = get_proteoform(proteoform_file, prsm_dir)
            assign_rt(proteoforms_original, ms1_features)
            proteoforms = remove_duplicate_proteoforms(proteoforms_original)
            proteins = set([p.accession for p in proteoforms])

            DIA_target_file = "20231117_DIA_" + mass_range + "_rep" + str(replicate)
            proteoform_file = os.path.join(file_dir, DIA_target_file + '_pseudo_ms2_toppic_proteoform_single.tsv')
            msalign_file = os.path.join(file_dir, DIA_target_file + "_pseudo_ms2.msalign")
            inte_dict = get_proteoform_inte_dict(msalign_file)
            prsm_dir = os.path.join(file_dir, DIA_target_file + "_pseudo_html\prsms")
            ms1_feature_file = os.path.join(os.path.join(file_dir, "ms1_features_mzrt"), DIA_target_file + "_frac_ms1.mzrt.csv")
            ms1_features = get_features(ms1_feature_file)
            DIA_proteoforms_original = get_proteoform(proteoform_file, prsm_dir, inte_dict, "DIA")
            assign_rt(DIA_proteoforms_original, ms1_features)
            DIA_proteoforms = remove_duplicate_proteoforms(DIA_proteoforms_original)
            DIA_proteins = set([p.accession for p in DIA_proteoforms])

            common = len(DIA_proteins.intersection(proteins))
            common_proteoforms = get_common_proteoforms(DIA_proteoforms, proteoforms)

            all_dda_proteins.append(len(proteins))
            all_dia_proteins.append(len(DIA_proteins))
            all_common_proteins.append(common)

            all_dda_proteoforms.append(len(proteoforms))
            all_dia_proteoforms.append(len(DIA_proteoforms))
            all_common_proteoforms.append(len(common_proteoforms))
            print("Pseudo - Proteoforms:", len(DIA_proteoforms_original),
                  "- Filtered Proteoforms:", len(DIA_proteoforms), "- Proteins:", len(DIA_proteins))
            print("TopPIC - Proteoforms:", len(proteoforms_original), "- Filtered Proteoforms:", len(proteoforms), "- Proteins:", len(proteins))
            print("TopPIC - Pseudo: common proteoforms", len(common_proteoforms), "- common proteins", common, "\n")

            # matched peaks
            DIA_matched_peaks_temp, DDA_matched_peaks_temp, DIA_num_peaks_temp, DDA_num_peaks_temp = get_matched_peaks(common_proteoforms)
            DIA_matched_peaks_shared.append(DIA_matched_peaks_temp)
            DDA_matched_peaks_shared.append(DDA_matched_peaks_temp)
            DIA_num_peaks_shared.append(DIA_num_peaks_temp)
            DDA_num_peaks_shared.append(DDA_num_peaks_temp)

            # peaks intensity - shared vs unique
            DIA_common_proteoforms_inte = [math.log2(pair[0].intensity) for pair in common_proteoforms]
            DDA_common_proteoforms_inte = [math.log2(pair[1].intensity) for pair in common_proteoforms]
            DIA_inte_common.append(DIA_common_proteoforms_inte)
            DDA_inte_common.append(DDA_common_proteoforms_inte)
            DIA_unique_proteoforms = [p for p in DIA_proteoforms if p.intensity not in DIA_common_proteoforms_inte]
            DDA_unique_proteoforms = [p for p in proteoforms if p.intensity not in DDA_common_proteoforms_inte]
            DIA_unique_proteoforms_inte = [math.log2(pair.intensity) for pair in DIA_unique_proteoforms]
            DDA_unique_proteoforms_inte = [math.log2(pair.intensity) for pair in DDA_unique_proteoforms]
            DIA_inte_unique.append(DIA_unique_proteoforms_inte)
            DDA_inte_unique.append(DDA_unique_proteoforms_inte)

            DDA_proteoforms_data.extend(proteoforms_original)
            DIA_proteoforms_data.extend(DIA_proteoforms_original)

        # For combined data
        DIA_proteoforms_data = remove_duplicate_proteoforms(DIA_proteoforms_data)
        DDA_proteoforms_data = remove_duplicate_proteoforms(DDA_proteoforms_data)
        DIA_proteins = set([p.accession for p in DIA_proteoforms_data])
        DDA_proteins = set([p.accession for p in DDA_proteoforms_data])
        print("Pseudo Proteoforms:", len(DIA_proteoforms_data), "- Proteins:", len(DIA_proteins))
        print("TopPIC Proteoforms:", len(DDA_proteoforms_data), "- Proteins:", len(DDA_proteins))

        common_proteoforms = get_common_proteoforms(DIA_proteoforms_data, DDA_proteoforms_data)
        dda_unique_prot = len(DDA_proteoforms_data) - len(common_proteoforms)
        dia_unique_prot = len(DIA_proteoforms_data) - len(common_proteoforms)
        common = len(DIA_proteins.intersection(DDA_proteins))
        common_prot = len(common_proteoforms)
        dda_unique = len(DDA_proteins) - common
        dia_unique = len(DIA_proteins) - common
        print("TopPIC - Pseudo: common proteoforms", len(common_proteoforms), round((len(common_proteoforms)/len(DDA_proteoforms_data)) * 100, 3))
        print("TopPIC - Pseudo: common proteins", common, round((common/len(DDA_proteins)) * 100, 3))

        # #############
        # matched peaks
        DIA_matched_peaks_temp, DDA_matched_peaks_temp, DIA_num_peaks_temp, DDA_num_peaks_temp = get_matched_peaks(common_proteoforms)
        DIA_matched_peaks_shared_num = sum([len(i) for i in DIA_matched_peaks_temp])
        DIA_matched_peaks_shared_len = len([len(i) for i in DIA_matched_peaks_temp])
        DDA_matched_peaks_shared_num = sum([len(i) for i in DDA_matched_peaks_temp])
        DDA_matched_peaks_shared_len = len([len(i) for i in DDA_matched_peaks_temp])

        DIA_num_peaks_shared_num = sum(DIA_num_peaks_temp)
        DIA_num_peaks_shared_len = len(DIA_num_peaks_temp)
        DDA_num_peaks_shared_num = sum(DDA_num_peaks_temp)
        DDA_num_peaks_shared_len = len(DDA_num_peaks_temp)

        dda_matched_peaks_per_scan = round(DDA_matched_peaks_shared_num/DDA_matched_peaks_shared_len, 3)
        dia_matched_peaks_per_scan = round(DIA_matched_peaks_shared_num/DIA_matched_peaks_shared_len, 3)

        dda_peaks_per_scan = round(DDA_num_peaks_shared_num/DDA_num_peaks_shared_len, 3)
        dia_peaks_per_scan = round(DIA_num_peaks_shared_num/DIA_num_peaks_shared_len, 3)

        print("Matched Peaks per common Scan (TopPIC vs Pseudo):", dda_matched_peaks_per_scan, dia_matched_peaks_per_scan)
        print("Num Peaks per common Scan (TopPIC vs Pseudo):", dda_peaks_per_scan, dia_peaks_per_scan)
        print("Ratio Peaks per common Scan (TopPIC vs Pseudo):", round(dda_matched_peaks_per_scan /
              dda_peaks_per_scan*100, 3), round(dia_matched_peaks_per_scan/dia_peaks_per_scan*100, 3), "\n")

        # 2d matched peaks plot
        DIA_matched_peaks_temp, DDA_matched_peaks_temp, DIA_num_peaks_temp, DDA_num_peaks_temp = get_matched_peaks(common_proteoforms)
        DIA_matched_peaks_shared_num = [len(i) for i in DIA_matched_peaks_temp]
        DDA_matched_peaks_shared_num = [len(i) for i in DDA_matched_peaks_temp]
        plt.Figure()
        sns.regplot(x='Number of Peaks', y='Number of Matched Peaks', data=pd.DataFrame(
            {'Number of Peaks': DIA_num_peaks_temp, 'Number of Matched Peaks': DIA_matched_peaks_shared_num}), marker='.', label="DIA")
        sns.regplot(x='Number of Peaks', y='Number of Matched Peaks', data=pd.DataFrame(
            {'Number of Peaks': DDA_num_peaks_temp, 'Number of Matched Peaks': DDA_matched_peaks_shared_num}), marker='.', label="DDA")
        plt.legend()
        # plt.savefig("01_matched_peaks_2d_toppic_" + str(replicate) + ".jpg", dpi=2000)
        plt.show()
        plt.close()

        all_dda_proteins_data.append(all_dda_proteins)
        all_dia_proteins_data.append(all_dia_proteins)
        all_common_proteins_data.append(all_common_proteins)
        all_dda_proteoforms_data.append(all_dda_proteoforms)
        all_dia_proteoforms_data.append(all_dia_proteoforms)
        all_common_proteoforms_data.append(all_common_proteoforms)

        dda_unique_proteoform_data.append(dda_unique_prot)
        dia_unique_proteoform_data.append(dia_unique_prot)
        common_proteoform_data.append(common_prot)
        dda_unique_protein_data.append(dda_unique)
        dia_unique_protein_data.append(dia_unique)
        common_protein_data.append(common)

    # ###################
    mean_dda_proteins_data = []
    mean_dia_proteins_data = []
    mean_common_proteins_data = []
    mean_dda_proteoforms_data = []
    mean_dia_proteoforms_data = []
    mean_common_proteoforms_data = []

    stddev_dda_proteins_data = []
    stddev_dia_proteins_data = []
    stddev_common_proteins_data = []
    stddev_dda_proteoforms_data = []
    stddev_dia_proteoforms_data = []
    stddev_common_proteoforms_data = []

    for i in range(0, len(mass_ranges)):
        mean_dia_proteins_data.append(mean([all_dia_proteins_data[0][i], all_dia_proteins_data[1][i]]))
        mean_dda_proteins_data.append(mean([all_dda_proteins_data[0][i], all_dda_proteins_data[1][i]]))
        mean_common_proteins_data.append(mean([all_common_proteins_data[0][i], all_common_proteins_data[1][i]]))

        mean_dia_proteoforms_data.append(mean([all_dia_proteoforms_data[0][i], all_dia_proteoforms_data[1][i]]))
        mean_dda_proteoforms_data.append(mean([all_dda_proteoforms_data[0][i], all_dda_proteoforms_data[1][i]]))
        mean_common_proteoforms_data.append(mean([all_common_proteoforms_data[0][i], all_common_proteoforms_data[1][i]]))

        stddev_dda_proteins_data.append(stdev([all_dda_proteins_data[0][i], all_dda_proteins_data[1][i]]))
        stddev_dia_proteins_data.append(stdev([all_dia_proteins_data[0][i], all_dia_proteins_data[1][i]]))
        stddev_common_proteins_data.append(stdev([all_common_proteins_data[0][i], all_common_proteins_data[1][i]]))

        stddev_dda_proteoforms_data.append(stdev([all_dda_proteoforms_data[0][i], all_dda_proteoforms_data[1][i]]))
        stddev_dia_proteoforms_data.append(stdev([all_dia_proteoforms_data[0][i], all_dia_proteoforms_data[1][i]]))
        stddev_common_proteoforms_data.append(stdev([all_common_proteoforms_data[0][i], all_common_proteoforms_data[1][i]]))

    ############################
    width = 0.3
    r = np.arange(len(mass_ranges))
    fig, axes = plt.subplots(2, 1, sharey=False, sharex=True)
    # Plot for Proteoforms
    axes[0].bar(r, mean_dia_proteoforms_data, color='r', alpha=0.5, width=width, edgecolor='black', label='DIA')
    axes[0].bar(r + width, mean_dda_proteoforms_data, color='g', alpha=0.5, width=width, edgecolor='black', label='DDA')
    axes[0].bar(r + width*2, mean_common_proteoforms_data, color='orange', alpha=0.5, width=width, edgecolor='black', label='Shared')
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].xaxis.set_tick_params(labelbottom=False)
    axes[0].set_ylabel('Proteoforms')
    # Plot for Proteins
    axes[1].bar(r, mean_dia_proteins_data, color='r', alpha=0.5, width=width, edgecolor='black', label='DIA')
    axes[1].bar(r + width, mean_dda_proteins_data, color='g', alpha=0.5, width=width, edgecolor='black', label='DDA')
    axes[1].bar(r + width*2, mean_common_proteins_data, color='orange', alpha=0.5, width=width, edgecolor='black', label='Shared')
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].xaxis.set_tick_params(labelbottom=True)
    axes[1].set_ylabel('Proteins')
    axes[1].set_xticks(r + width)
    axes[1].set_xticklabels(['[720-800]', '[800-880]', '[880-960]', '[960-1040]', '[1040-1120]', '[1120-1200]'], rotation=25)
    # Add Combined Legend on Top
    combined_legend_labels = ['Pseudo spectra', 'Single spectra', 'Shared']
    fig.legend(combined_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(combined_legend_labels))
    plt.subplots_adjust(bottom=0.175)
    # plt.savefig("000_combined_frequency_toppic_" + str(replicate) + ".jpg", dpi=2000, bbox_inches='tight')
    plt.show()
    plt.close()

    ############################

    plt.Figure()
    out = venn2(subsets=(dia_unique_proteoform_data[0], dda_unique_proteoform_data[0],
                common_proteoform_data[0]), set_labels=('Pseudo', 'Single'))
    for text in out.set_labels:
        text.set_fontsize(14)
    for text in out.subset_labels:
        text.set_fontsize(16)
    plt.title('Proteoforms', fontsize=16)
    # plt.savefig("01_venn_Proteoforms_toppic_" + str(replicates[0]) + ".jpg", dpi=2000, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure()
    out = venn2(subsets=(dia_unique_proteoform_data[1], dda_unique_proteoform_data[1],
                common_proteoform_data[1]), set_labels=('Pseudo', 'Single'))
    for text in out.set_labels:
        text.set_fontsize(14)
    for text in out.subset_labels:
        text.set_fontsize(16)
    plt.title('Proteoforms', fontsize=16)
    # plt.savefig("01_venn_Proteoforms_toppic_" + str(replicates[1]) + ".jpg", dpi=2000, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure()
    out = venn2(subsets=(dia_unique_protein_data[0], dda_unique_protein_data[0],
                common_protein_data[0]), set_labels=('Pseudo', 'Single'))
    for text in out.set_labels:
        text.set_fontsize(14)
    for text in out.subset_labels:
        text.set_fontsize(16)
    plt.title('Proteins', fontsize=16)
    # plt.savefig("01_venn_Proteins_toppic_" + str(replicates[0]) + ".jpg", dpi=2000, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure()
    out = venn2(subsets=(dia_unique_protein_data[1], dda_unique_protein_data[1],
                common_protein_data[1]), set_labels=('Pseudo', 'Single'))
    for text in out.set_labels:
        text.set_fontsize(14)
    for text in out.subset_labels:
        text.set_fontsize(16)
    plt.title('Proteins', fontsize=16)
    # plt.savefig("01_venn_Proteins_toppic_" + str(replicates[1]) + ".jpg", dpi=2000, bbox_inches='tight')
    plt.show()
    plt.close()
