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

def get_mz(mass, charge):
    proton = 1.00727
    return (mass + (charge * proton)) / charge


class Proteoform:
    def __init__(self, i, spec_idx, idx, rt, rt_apex, charge, mass, sequence, acetylation, score, num_peaks, matched_peaks, intensity, accession=None, evalue=None):
        self.i = i
        self.spec_idx = spec_idx
        self.idx = idx
        self.rt = rt
        self.rt_apex = rt_apex
        self.charge = charge
        self.mass = mass
        self.sequence = sequence
        self.acetylation = acetylation
        self.mz = get_mz(mass, charge)
        self.score = score
        self.num_peaks = num_peaks
        self.matched_peaks = matched_peaks
        self.intensity = intensity
        self.accession = accession
        self.evalue = evalue
        self.prefix_ions = None
        self.suffix_ions = None

    def to_dict(self):
        return {
            'i': self.i,
            'spec_idx': self.spec_idx,
            'idx': self.idx,
            'rt': self.rt,
            'charge': self.charge,
            'mass': self.mass,
            'adjusted_mass': self.adjusted_mass,
            'mz': self.mz,
            'accession': self.accession,
            'sequence': self.sequence,
            'num_peaks': self.num_peaks,
            'matched_peaks': self.matched_peaks,
            'intensity': self.intensity,
            'E-value': self.evalue,
            'first_residue': self.first_residue,
            'last_residue': self.last_residue
        }
