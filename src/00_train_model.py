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

#!/usr/bin/env python3

import os
import math
import json
import pymzml
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from proteoform import Proteoform
from read_msalign import read_spec_file
from feature import Feature


def get_rt_dictionary(mzML_file, isolation_window_base):
    run = pymzml.run.Reader(mzML_file, MS1_Precision=5e-6, MSn_Precision=20e-6, skip_chromatogram=True)
    isolation_window_rts = [[] for _ in range(len(isolation_window_base) + 1)]
    for s in run:
        if s.ms_level == 1:
            isolation_window_rts[0].append(s.scan_time[0])
        else:
            iso_window = isolation_window_base.index(s.get("MS:1000827", -1))
            if iso_window != -1:
                isolation_window_rts[iso_window + 1].append(s.scan_time[0])
    rt_ms1 = isolation_window_rts[0]
    for idx in range(1, len(isolation_window_rts)):
        if len(isolation_window_rts[idx]) < len(rt_ms1):
            isolation_window_rts[idx].append(rt_ms1[-1])
    return isolation_window_rts


def get_mz(mass, charge):
    return (mass + (charge * 1.00727)) / charge


def get_features(feature_file):
    df = pd.read_csv(feature_file)
    features = []
    for i in range(0, len(df)):
        data = df.iloc[i]
        score = data['Score']
        if math.isnan(score):
            score = 0
        xic = []
        if 'XIC' in df.keys():
            split_xic = data['XIC'].split(';')
            xic = [float(v) for v in split_xic]
        envelope = []
        if 'Envelope' in df.keys():
            split_Envelope = data['Envelope'].split(';')
            envelope = [(float(peak.split('&')[0]), float(peak.split('&')[1])) for peak in split_Envelope]
        rt_apex = 0
        if 'Envelope' in df.keys():
            rt_apex = data['rtApex']
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


def get_proteoform(proteoform_file, prsm_dir):
    df = pd.read_csv(proteoform_file, skiprows=35, delimiter="\t")
    proteoforms = []
    for i in range(0, len(df)):
        data = df.iloc[i]
        spectrum_id = -1
        if 'Spectrum ID' in data:
            spectrum_id = data['Spectrum ID']
        scan = -1
        if 'Scan(s)' in data:
            scan = data['Scan(s)']
        prsm_id = -1
        if 'Prsm ID' in data:
            prsm_id = data['Prsm ID']
        rt = -1
        if 'Retention time' in data:
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
        accession = ""
        if 'Protein accession' in data:
            accession = data['Protein accession']
        score = -1
        if 'Feature score' in data:
            score = data['Feature score']
        evalue = -1
        if 'E-value' in data:
            evalue = data['E-value']
        unexpected_mods = 0
        if '#unexpected modifications' in data:
            unexpected_mods = data['#unexpected modifications']
        var_mods = 0
        if '#variable PTMs' in data:
            var_mods = data['#variable PTMs']
        feature_id = 0
        if 'Feature ID' in data:
            feature_id = data['Feature ID']
        if var_mods != 0 or unexpected_mods != 0 or evalue > 0.01:
            continue
        acetylation = False
        proteoform = Proteoform(i, spectrum_id, prsm_id, rt, apex_rt, charge, mass, seq, acetylation,
                                score, peak, matched_fragment, feature_intensity, accession, evalue)
        if prsm_dir is not None:
            prsm_file = os.path.join(prsm_dir, "prsm" + str(prsm_id) + ".js")
            matched_peaks = return_fragments(prsm_file)
            proteoform.matched_peaks = matched_peaks
            proteoform.feature_id = feature_id
            proteoform.scan = scan
        proteoforms.append(proteoform)
    return proteoforms


def assign_rt(proteoforms, ms1_features):
    ms1_features_identified = []
    proteoforms_with_features = []
    for p in proteoforms:
        for mf in ms1_features:
            if abs(round(mf.mass, 3) - round(p.mass, 3)) < 0.1 and mf.charge == p.charge and round(mf.rt_low, 2) <= p.rt/60 <= round(mf.rt_high, 2):
                p.rt_low = mf.rt_low
                p.rt_high = mf.rt_high
                p.rt_apex = mf.rt_apex
                p.mz = get_mz(p.mass, p.charge)
                p.envelope = mf.envelope
                ms1_features_identified.append(mf)
                proteoforms_with_features.append(p)
                break
    return ms1_features_identified, proteoforms_with_features


def map_matched_frags(spec, p):
    match_counter = 0
    for m in spec.peak_list:
        for p_m in p.matched_peaks:
            if (abs(m.mass - p_m[0]) < 0.01 and m.charge == p_m[1] and abs(m.intensity - p_m[2]) < 0.01):
                m.matched = 1
                match_counter += 1
                break


def _getExtendMasses(mass):
    IM = 1.00235
    extend_offsets_ = [0, -IM, IM, 2 * -IM, 2 * IM]
    result = []
    for i in range(0, len(extend_offsets_)):
        new_mass = mass + extend_offsets_[i]
        result.append(new_mass)
    return result


def get_apex_cycle_distance(ms1_feature, ms2_feature):
    ms1_apex_cycle = ms1_feature.xic.index(max(ms1_feature.xic))
    ms2_apex_cycle = ms2_feature.xic.index(max(ms2_feature.xic))
    apex_cycle_distance = abs(ms1_apex_cycle - ms2_apex_cycle)
    return apex_cycle_distance


def normalize_xic(xic):
    total_sum = sum(xic)
    normalized_xic = [value / total_sum for value in xic]
    return normalized_xic


def interp(x, xp, fp):
    interpolated_values = []
    for x_val in x:
        i = 0
        while i < len(xp) and xp[i] < x_val:
            i += 1

        if i == 0 or i == len(xp):
            interpolated_values.append(0.0)
            continue

        x1 = xp[i - 1]
        x2 = xp[i]
        y1 = fp[i - 1]
        y2 = fp[i]
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x_val - x1)
        interpolated_values.append(y)

    return interpolated_values


def get_overlap(ms1_feature, ms2_feature):
    start_rt = min(ms1_feature.rt_low, ms2_feature.rt_low)
    end_rt = max(ms1_feature.rt_high, ms2_feature.rt_high)
    return (start_rt, end_rt)


def replace_nan(lst):
    return [0 if math.isnan(x) else x for x in lst]

# Get the pseudo spectrum


class SpecPeaks:
    def __init__(self, mass, mono_mz, charge, intensity, apex_scan_diff, shared_area, corr):
        self.mass = mass
        self.mono_mz = mono_mz
        self.charge = charge
        self.intensity = intensity
        self.apex_scan_diff = apex_scan_diff
        self.shared_area = shared_area
        self.corr = corr


def get_isolation_window_ms1_features_maxInte(ms1_features, isolation_window_base, isolation_window_base_index):
    indices = []
    isolation_window_half = 2  # half-width of the isolation window
    target_window = isolation_window_base[isolation_window_base_index]
    for feature_idx, ms1_feature in enumerate(ms1_features):
        envelope_intensity = sum(peak[1] for peak in ms1_feature.envelope)
        if envelope_intensity > 0:
            max_coverage = 0
            selected_window = -1
            for window in isolation_window_base:
                window_start = window - isolation_window_half
                window_end = window + isolation_window_half
                assigned_intensity = sum(peak[1] for peak in ms1_feature.envelope if window_start <= peak[0] < window_end)
                coverage = assigned_intensity / envelope_intensity
                if coverage > max_coverage:
                    max_coverage = coverage
                    selected_window = window
            if selected_window == target_window and sum(ms1_feature.xic) > 0:
                indices.append(feature_idx)
    return indices


def interpolate_xic(xic, rt, rt_target):
    xic = np.append(xic, np.zeros(max(0, len(rt) - len(xic))))
    feature_xic_ms2_interp = np.interp(rt_target, rt, xic)
    xic_interp = np.where(np.logical_or(np.isnan(feature_xic_ms2_interp), np.isinf(feature_xic_ms2_interp)), 0, feature_xic_ms2_interp,)
    return xic_interp


def compute_shared_area(ms1_feature, ms2_feature, rt_ms1, rt_ms2, rt_target):
    ms1_xic = interpolate_xic(ms1_feature.xic[0:len(rt_ms2)], rt_ms1[0:len(rt_ms2)], rt_target)
    ms2_xic = interpolate_xic(ms2_feature.xic[0:len(rt_ms2)], rt_ms2, rt_target)
    ms1_xic = normalize_xic(ms1_xic)
    ms2_xic = normalize_xic(ms2_xic)
    shared_area = 0.0
    for i in range(len(ms1_xic)):
        shared_area += min(ms1_xic[i], ms2_xic[i])
    return shared_area


def generate_msalign(ms1_features, ms2_data, rt_features, isolation_windows):
    max_value = max([max(i) for i in rt_features])
    interval = 0.01
    rt_target = np.arange(0, max_value + interval, interval)
    assigned_ms2 = [[] for _ in range(len(ms1_features))]
    for isolation_windows_index in range(len(isolation_windows)):
        rt_ms1 = rt_features[0]
        rt_ms2 = rt_features[isolation_windows_index+1]
        if len(rt_ms2) < len(rt_ms1):
            rt_ms2.append(rt_ms1[-1])
        print("Processing Isolation window:", isolation_windows_index)
        base_mz = isolation_windows[isolation_windows_index]
        ms2_features = ms2_data[str(base_mz)]
        indices = get_isolation_window_ms1_features_maxInte(ms1_features, isolation_windows, isolation_windows_index)
        for ms2_feature_idx in range(0, len(ms2_features)):
            if ms2_feature_idx % 1000 == 0:
                print("Processing MS2 Feature:", ms2_feature_idx)
            ms2_feature = ms2_features[ms2_feature_idx]
            for feature_idx in indices:
                ms1_feature = ms1_features[feature_idx]
                ms1_features[feature_idx].base_mz = isolation_windows[isolation_windows_index]
                length_ms1_feature = sum(bool(x) for x in ms1_feature.xic)
                if ms2_feature.mass < ms1_feature.mass:
                    apex_cycle_distance = get_apex_cycle_distance(ms1_feature, ms2_feature)
                    # apex_cycle_distance_tol = min(3, length_ms1_feature/2)
                    apex_cycle_distance_tol = int(length_ms1_feature/2)
                    if apex_cycle_distance > apex_cycle_distance_tol:
                        continue

                    length_ms2_feature = sum(bool(x) for x in ms2_feature.xic)
                    shared_area = compute_shared_area(ms1_feature, ms2_feature, rt_ms1, rt_ms2, rt_target)
                    peak = SpecPeaks(ms2_feature.mass, ms2_feature.mono_mz, ms2_feature.charge,
                                     ms2_feature.intensity, apex_cycle_distance, shared_area, 0)
                    peak.rt_low = ms2_feature.rt_low
                    peak.rt_high = ms2_feature.rt_high
                    peak.length = length_ms2_feature
                    assigned_ms2[feature_idx].append(peak)
    return assigned_ms2


def plot_apex_distance_distribution():
  # generate Rank plot
    all_frag_list = []
    ms1_features = []
    proteoforms = []
    counter = 0
    for file_idx in range(0, len(proteoform_files)):
        pairs_1d = pseudo_data[file_idx][0]
        assigned_ms2_data = pseudo_data[file_idx][1]
        frag_list = []
        for p_idx in range(0, len(assigned_ms2_data)):
            frags = []
            for pseudo_peak in assigned_ms2_data[p_idx]:
                frags.append(pseudo_peak)
                counter += 1
            frag_list.append(frags)
            ms1_features.append(pairs_1d[p_idx][1])
            proteoforms.append(pairs_1d[p_idx][0])
        all_frag_list.append(frag_list)

    pseudo_spectra = []
    max_rank_len = 0
    for frag_list in all_frag_list:
        rank_len = max([len(frags) for frags in frag_list])
        pseudo_spectra.extend(frag_list)
        if max_rank_len < rank_len:
            max_rank_len = rank_len

    labels = []
    X_inte_rank_ratio = []
    for spec_id in range(0, len(pseudo_spectra)):
        pseudo_spectrum = pseudo_spectra[spec_id]
        pseudo_spectrum = sorted(pseudo_spectrum, key=lambda x: x.intensity, reverse=True)
        rank = len(pseudo_spectrum)
        for ms2_feature in pseudo_spectrum:
            labels.append(ms2_feature.label)
            X_inte_rank_ratio.append((ms2_feature.apex_scan_diff))
            rank -= 1

    X = X_inte_rank_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=labels)
    class_weights = {0: class_weights[0], 1: class_weights[1]}

    ########################################
    ##############################
    positive_apex_distance = []
    for i in range(0, len(y_test)):
        apex_distance = X_test[i]
        label = y_test[i]
        if label == 1:
            positive_apex_distance.append(apex_distance)
    values = list(range(max(positive_apex_distance)))
    freq_list = [positive_apex_distance.count(value) for value in values]
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
    combined_values = freq_list[:9] + [sum(freq_list[9:])]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(x)), combined_values)
    plt.xticks(range(len(x)), x)
    plt.xlabel('Apex Cycle Distance')
    plt.ylabel('Number of Positive Feature Pairs')
    plt.show()
    plt.close()


if __name__ == '__main__':
    train_data_dir = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "TopDIA_Published_Data"), r"03_train_model")
    mz_ranges = ['720_800', '800_880', '880_960', '960_1040', '1040_1120', '1120_1200']
    isolation_window_base = [[722, 726, 730, 734, 738, 742, 746, 750, 754, 758, 762, 766, 770, 774, 778, 782, 786, 790, 794, 798],
                             [802, 806, 810, 814, 818, 822, 826, 830, 834, 838, 842, 846, 850, 854, 858, 862, 866, 870, 874, 878],
                             [882, 886, 890, 894, 898, 902, 906, 910, 914, 918, 922, 926, 930, 934, 938, 942, 946, 950, 954, 958],
                             [962, 966, 970, 974, 978, 982, 986, 990, 994, 998, 1002, 1006, 1010, 1014, 1018, 1022, 1026, 1030, 1034, 1038],
                             [1042, 1046, 1050, 1054, 1058, 1062, 1066, 1070, 1074, 1078, 1082, 1086, 1090, 1094, 1098, 1102, 1106, 1110, 1114, 1118],
                             [1122, 1126, 1130, 1134, 1138, 1142, 1146, 1150, 1154, 1158, 1162, 1166, 1170, 1174, 1178, 1182, 1186, 1190, 1194, 1198]]
    proteoform_files = ["20231117_DIA_" + mz_range + "_rep1_ms2_toppic_proteoform_single.tsv" for mz_range in mz_ranges]
    mzml_files = ["20231117_DIA_" + mz_range + "_rep1.mzml" for mz_range in mz_ranges]

    # Load pseudo spectra
    print("*** Loading Pseudo Data ***")
    file_path = os.path.join(train_data_dir, 'pseudo_data.pickle')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            pseudo_data = pickle.load(file)
    else:
        # Read the data files
        print("Reading Data Files")
        rt_dictionary_list = []
        proteoforms_window = []
        ms1_data_window = []
        ms2_data_window = []
        ms2_spec_list = []
        for file_idx in range(0, len(proteoform_files)):
            mzml_file = os.path.join(os.path.join(train_data_dir, "mzml_files"), mzml_files[file_idx])
            rt_dictionary = get_rt_dictionary(mzml_file, isolation_window_base[file_idx])
            rt_dictionary_list.append(rt_dictionary)

            feature_file = os.path.join(os.path.join(train_data_dir, "ms1_features_mzrt"), mzml_files[file_idx][0:-5] + "_frac_ms1.mzrt.csv")
            ms1_features = get_features(feature_file)

            proteoform_file = os.path.join(train_data_dir, proteoform_files[file_idx])
            prsm_dir = os.path.join(train_data_dir, mzml_files[file_idx][0:-5] + "_html\prsms")
            proteoforms = get_proteoform(proteoform_file, prsm_dir)
            ms1_features_identified, proteoforms = assign_rt(proteoforms, ms1_features)
            ms1_data_window.append(ms1_features_identified)
            proteoforms_window.append(proteoforms)

            msalign_file = os.path.join(train_data_dir, mzml_files[file_idx][0:-5] + "_ms2.msalign")
            ms2_spectra = read_spec_file(msalign_file)
            ms2_spec_list.append(ms2_spectra)

            isolation_window = isolation_window_base[file_idx]
            ms2_data = {}
            for i in range(0, len(isolation_window)):
                ms2_feature_file = os.path.join(train_data_dir, mzml_files[file_idx][0:-5] + "_ms2_features\\" + mzml_files[file_idx][0:-5] + "_" +
                                                str(isolation_window[i]-2) + "_" + str(isolation_window[i]+2) + "_frac_ms2.mzrt.csv")
                ms2_features = get_features(ms2_feature_file)
                ms2_data[str(isolation_window[i])] = ms2_features
            ms2_data_window.append(ms2_data)

        #########################################################################################################
        #########################################################################################################
        # Load pseudo spectra
        print("Generating Pseudo Data")
        data = []
        for file_idx in range(0, len(proteoform_files)):
            # Read data
            proteoforms = proteoforms_window[file_idx]
            ms1_features_identified = ms1_data_window[file_idx]
            ms2_data = ms2_data_window[file_idx]
            rt_dictionary = rt_dictionary_list[file_idx]
            ms2_spectra = ms2_spec_list[file_idx]
            fragment_feature_pair = []
            for p_idx in range(0, len(proteoforms)):
                p = proteoforms[p_idx]
                ms1_feature = ms1_features_identified[p_idx]
                # get fragment peaks in the proteoform scan
                spec = [s for s in ms2_spectra if s.header.spec_scan == p.scan][0]
                map_matched_frags(spec, p)
                # map fragments to ms2 features
                ms2_features = ms2_data[str(int((spec.header.prec_win_begin + spec.header.prec_win_end)/2))]
                shortlisted_features = []
                error_tole = 10E-6*p.mass
                for m in spec.peak_list:
                    ext_masses = _getExtendMasses(m.mass)
                    rt = spec.header.retention_time/60
                    for f in ms2_features:
                        if abs(rt - f.rt_apex) < 0.5 and f.charge == m.charge:
                            for k in range(0, len(ext_masses)):
                                mass_diff = abs(ext_masses[k] - f.mass)
                                if (mass_diff <= error_tole):
                                    f.label = m.matched
                                    shortlisted_features.append(f)
                                    break
                fragment_feature_pair.append((p, ms1_feature, shortlisted_features))
            data.append(fragment_feature_pair)

        # Generating pseudo spectra
        pseudo_data = []
        for file_idx in range(0, len(proteoform_files)):
            fragment_feature_pair = data[file_idx]
            # get isolation window data
            ms1_features = []
            pairs_1d = []
            for p_idx in range(0, len(fragment_feature_pair)):
                proteoform = fragment_feature_pair[p_idx][0]
                ms1_feature = fragment_feature_pair[p_idx][1]
                shortlisted_features = fragment_feature_pair[p_idx][2]
                ms1_features.append(ms1_feature)
                pairs_1d.append((proteoform, ms1_feature, shortlisted_features))
            # get isolation window data
            ms2_data = ms2_data_window[file_idx]
            rt_features = rt_dictionary_list[file_idx]
            isolation_windows = isolation_window_base[file_idx]
            assigned_ms2_data = generate_msalign(ms1_features, ms2_data, rt_features, isolation_windows)
            pseudo_data.append((pairs_1d, assigned_ms2_data))

        with open(os.path.join(train_data_dir, 'pseudo_data.pickle'), 'wb') as file:
            pickle.dump(pseudo_data,  file)

    # pseudo_data_bk_interpolated = deepcopy(pseudo_data)
    #########################################################################################################
    # Assign labels
    for file_idx in range(0, len(proteoform_files)):
        pairs_1d = pseudo_data[file_idx][0]
        assigned_ms2_data = pseudo_data[file_idx][1]
        for p_idx in range(0, len(pairs_1d)):
            p = pairs_1d[p_idx][0]
            ms1_feature = pairs_1d[p_idx][1]
            shortlisted_features = pairs_1d[p_idx][2]
            pseudo_features = assigned_ms2_data[p_idx]
            sorted(pseudo_features, key=lambda x: x.length, reverse=True)
            # map fragments to pseudo ms2 features
            error_tole = 10E-6*p.mass
            for m in pseudo_features:
                m.label = 0
                ext_masses = _getExtendMasses(m.mass)
                for p_m in p.matched_peaks:
                    if m.charge == p_m[1]:
                        for k in range(0, len(ext_masses)):
                            mass_diff = abs(ext_masses[k] - p_m[0])
                            if (mass_diff <= error_tole):
                                m.label = 1
                                break

    plot_apex_distance_distribution()
    #########################################################################################################
    #########################################################################################################
    # Filter features by apex distance tolerance.
    for file_idx in range(0, len(proteoform_files)):
        pairs_1d = pseudo_data[file_idx][0]
        assigned_ms2_data = pseudo_data[file_idx][1]
        for p_idx in range(0, len(pairs_1d)):
            p = pairs_1d[p_idx][0]
            ms1_feature = pairs_1d[p_idx][1]
            shortlisted_features = pairs_1d[p_idx][2]
            pseudo_features = assigned_ms2_data[p_idx]
            sorted(pseudo_features, key=lambda x: x.length, reverse=True)
            length_ms1_feature = sum(bool(x) for x in ms1_feature.xic)
            apex_cycle_distance_tol = min(3, length_ms1_feature/2)
            for m in pseudo_features:
                if m.apex_scan_diff <= apex_cycle_distance_tol:
                    m.use = True
                else:
                    m.use = False

    #########################################################################################################
    #########################################################################################################
    # generate Rank plot
    print("*** Train/Test Model ***")
    all_frag_list = []
    ms1_features = []
    proteoforms = []
    counter = 0
    for file_idx in range(0, len(proteoform_files)):
        pairs_1d = pseudo_data[file_idx][0]
        assigned_ms2_data = pseudo_data[file_idx][1]
        frag_list = []
        for p_idx in range(0, len(assigned_ms2_data)):
            frags = []
            for pseudo_peak in assigned_ms2_data[p_idx]:
                if pseudo_peak.use == False:
                    continue
                frags.append(pseudo_peak)
                counter += 1
            frag_list.append(frags)
            ms1_features.append(pairs_1d[p_idx][1])
            proteoforms.append(pairs_1d[p_idx][0])
        all_frag_list.append(frag_list)
    print("Number of Proteoforms:", sum([len(i) for i in all_frag_list]))
    print("Precursor Fragment Feature Pairs:", counter)

    pseudo_spectra = []
    max_rank_len = 0
    for frag_list in all_frag_list:
        rank_len = max([len(frags) for frags in frag_list])
        pseudo_spectra.extend(frag_list)
        if max_rank_len < rank_len:
            max_rank_len = rank_len

    # GET ROC AUC -- # For long data
    labels = []
    X_inte_rank_ratio = []
    for spec_id in range(0, len(pseudo_spectra)):
        ms1_feature = ms1_features[spec_id]
        ms1_feature.length = sum(bool(x) for x in ms1_feature.xic)
        pseudo_spectrum = pseudo_spectra[spec_id]
        pseudo_spectrum = sorted(pseudo_spectrum, key=lambda x: x.intensity, reverse=True)
        rank = len(pseudo_spectrum)
        for ms2_feature in pseudo_spectrum:
            ms2_feature.inten_rank = rank
            ms2_feature.inten_rank_ratio = rank/len(pseudo_spectrum)
            if math.isnan(ms2_feature.corr):
                ms2_feature.corr = 0
            labels.append(ms2_feature.label)
            X_inte_rank_ratio.append((ms2_feature.inten_rank_ratio, ms2_feature.length/ms1_feature.length, ms2_feature.shared_area))
            rank -= 1

    XX_labels = "Intensity Rank Ratio"
    X = X_inte_rank_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=labels)
    class_weights = {0: class_weights[0], 1: class_weights[1]}

    ########################################
    model = LogisticRegression(random_state=42, class_weight=class_weights)
    clf_ApexCycleDist_Inte = model.fit(X_train, y_train)
    coef = clf_ApexCycleDist_Inte.coef_[0]

    preds = clf_ApexCycleDist_Inte.predict_proba(X_test)
    preds = [p[1] for p in preds]
    auc = roc_auc_score(y_test, preds)
    print('Model AUC: ' + str(round(auc, 4)))

    predictions = [0 if i < 0.5 else 1 for i in preds]
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    print('Model balanced accuracy: ' + str(round(balanced_accuracy, 4)))

    # X_test_Inte = [x[0] for x in X_test]
    # auc = roc_auc_score(y_test, X_test_Inte)
    # print('Inte AUC: ' + str(round(auc, 4)))

    model_ranksum = []
    model_rank = [0] * max_rank_len
    for spec_id in range(0, len(pseudo_spectra)):
        ms1_feature = ms1_features[spec_id]
        pseudo_spectrum = pseudo_spectra[spec_id]
        for ms2_feature in pseudo_spectrum:
            x = [(ms2_feature.inten_rank_ratio, ms2_feature.length/ms1_feature.length, ms2_feature.shared_area)]
            ms2_feature.score = clf_ApexCycleDist_Inte.predict_proba(x)[0][1]
        pseudo_spectrum = sorted(pseudo_spectrum, key=lambda x: x.score, reverse=True)
        spec_ranksum = 0
        for j in range(0, len(pseudo_spectrum)):
            if j >= max_rank_len:
                break
            if pseudo_spectrum[j].label:
                model_rank[j] = model_rank[j] + 1
                spec_ranksum = spec_ranksum + (j + 1)
        model_ranksum .append(spec_ranksum)
    print("Model RankSUM value:", sum(model_ranksum))

    inte_ranksum = []
    inte_rank = [0] * max_rank_len
    for spec_id in range(0, len(pseudo_spectra)):
        ms1_feature = ms1_features[spec_id]
        pseudo_spectrum = pseudo_spectra[spec_id]
        for ms2_feature in pseudo_spectrum:
            pseudo_spectrum = sorted(pseudo_spectrum, key=lambda x: x.inten_rank_ratio, reverse=True)
        spec_ranksum = 0
        for j in range(0, len(pseudo_spectrum)):
            if j >= max_rank_len:
                break
            if pseudo_spectrum[j].label:
                inte_rank[j] = inte_rank[j] + 1
                spec_ranksum = spec_ranksum + (j + 1)
                # spec_ranksum = spec_ranksum + (max_rank_len - j)
        inte_ranksum.append(spec_ranksum)
    # print("Intensity RankSUM value:", sum(inte_ranksum), "\n")

    # plt_rank_len = max_rank_len
    plt_rank_len = 200
    plt.figure()
    plt.plot(list(range(0, plt_rank_len)), inte_rank[0:plt_rank_len])
    plt.plot(list(range(0, plt_rank_len)), model_rank[0:plt_rank_len])
    plt.title(' Rank Plot ' + XX_labels)
    plt.ylabel('Number of PrSMs with label 1')
    plt.xlabel('Rank Number')
    plt.legend(['Intensity',  'Model'], loc='upper right')
    plt.show()
    # plt.savefig("low_mass_rank.png", dpi=500)
    plt.close()

    print("**************Model Weights**************")
    print("Intercept:", clf_ApexCycleDist_Inte.intercept_[0])
    coef = clf_ApexCycleDist_Inte.coef_[0]
    words = ['Intensity Ratio:', 'MS2 Feature Length:', 'Shared Area:']
    for i in range(len(words)):
        print(words[i] + " " + str(coef[i]))

    X_inte_rank_ratio_roc = [x[0] for x in X_test]
    fpr_1, tpr_1, threshold_1 = roc_curve(y_test, X_inte_rank_ratio_roc)
    preds = clf_ApexCycleDist_Inte.predict_proba(X_test)
    preds = [p[1] for p in preds]
    fpr_4, tpr_4, threshold_4 = roc_curve(y_test, preds)

    plt.plot(fpr_4, tpr_4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC curve for Logistic Regression Model")
    plt.show()
    # plt.savefig("ROC_logistic_regression_model.png", dpi=1500)
    plt.close()

    thr = threshold_4[np.argmin(abs(tpr_4-(1-fpr_4)))]
    plt.figure()
    plt.plot(threshold_4[1:], tpr_4[1:])
    plt.plot(threshold_4[1:], fpr_4[1:])
    plt.axvline(thr, ls='--', color='grey', alpha=0.5)
    plt.xlabel("Score")
    plt.ylabel("True and False Positive Rate")
    plt.legend(['True Positive Rate', 'False Positive Rate'])
    plt.title(' Thr: ' + str(round(thr, 3)))
    # plt.savefig("TP_FP_Rate.png", dpi=1200)
    plt.show()
    plt.close()
