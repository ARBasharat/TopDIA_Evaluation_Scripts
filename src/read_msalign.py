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

from spectrum import Spectrum, SpecHeader, SpecPeak


def __read_header(spec_lines):
    spec_id = -1
    spec_scan = -1
    retention_time = -1
    ms1_spec_id = -1
    ms1_spec_scan = -1
    prec_win_begin = -1
    prec_win_end = -1
    prec_mz = []
    prec_charge = []
    prec_mass = []
    prec_intensity = []
    for i in range(len(spec_lines)):
        line = spec_lines[i]
        mono = line.split('=')
        if ("SPECTRUM_ID" in line):
            spec_id = int(mono[1])
        if ("SCANS" in line):
            spec_scan = int(mono[1])
        if ("RETENTION_TIME" in line):
            retention_time = float(mono[1])
        if ("MS_ONE_ID" in line):
            ms1_spec_id = int(mono[1])
        if ("MS_ONE_SCAN" in line):
            ms1_spec_scan = int(mono[1])

        if ("PRECURSOR_WINDOW_BEGIN" in line):
            prec_win_begin = float(mono[1])
        if ("PRECURSOR_WINDOW_END" in line):
            prec_win_end = float(mono[1])

        if ('PRECURSOR_MZ=' != line):
            if ("PRECURSOR_MZ" in line):
                prec_list = mono[1].split(':')
                for mz in prec_list:
                    prec_mz.append(float(mz))
        if ('PRECURSOR_CHARGE=' != line):
            if ("PRECURSOR_CHARGE" in line):
                prec_list = mono[1].split(':')
                for charge in prec_list:
                    prec_charge.append(float(charge))
        if ('PRECURSOR_MASS=' != line):
            if ("PRECURSOR_MASS" in line):
                prec_list = mono[1].split(':')
                for mass in prec_list:
                    prec_mass.append(float(mass))
        if ('PRECURSOR_INTENSITY=' != line):
            if ("PRECURSOR_INTENSITY" in line):
                prec_list = mono[1].split(':')
                for inte in prec_list:
                    prec_intensity.append(float(inte))

    header = SpecHeader(spec_id, spec_scan, retention_time, ms1_spec_id,
                        ms1_spec_scan, prec_mz, prec_charge, prec_mass, prec_intensity)
    header.prec_win_begin = prec_win_begin
    header.prec_win_end = prec_win_end
    return header


def __read_peaks(spec_lines):
    exp_line = "PRECURSOR_FEATURE_ID"
    end_line = "END IONS"
    peak_list = []
    i = 0
    while (exp_line not in spec_lines[i]):
        i = i + 1
    i = i + 1
    while (spec_lines[i] != end_line):
        mono = spec_lines[i].split()
        mass = float(mono[0])
        intensity = float(mono[1])
        charge = int(mono[2])
        peak = SpecPeak(mass, intensity, charge)
        if len(mono) > 4:
            peak.apex_scan_diff = int(float(mono[3]))
            peak.shared_intensity = float(mono[4])
            peak.corr = float(mono[5])
        peak.matched = 0
        peak_list.append(peak)
        i = i + 1
    return peak_list


def __parse_spectrum(spec_lines):
    header = __read_header(spec_lines)
    peak_list = __read_peaks(spec_lines)
    spec = Spectrum.get_spec(header, peak_list)
    return spec


def __get_end_index(all_lines, begin_idx):
    idx = begin_idx
    while (idx < len(all_lines) and "END IONS" not in all_lines[idx]):
        idx = idx + 1
    return idx


def read_spec_file(filename):
    file = open(filename)
    all_lines = file.readlines()
    all_lines = [x.strip() for x in all_lines]
    file.close()
    # Assign file name to header
    spec_list = []
    begin_idx = 0
    while (begin_idx < len(all_lines)):
        end_idx = __get_end_index(all_lines, begin_idx)
        spec_lines = all_lines[begin_idx:end_idx + 1]
        begin_idx = end_idx + 1
        if begin_idx >= len(all_lines):
            break
        spec = __parse_spectrum(spec_lines)
        spec_list.append(spec)
    return spec_list
