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

class Feature:
    def __init__(self, idx, mass, mono_mz, charge, intensity, mz_low, mz_high, rt_low, rt_high, rt_apex, ecscore, xic, envelope=None):
        self.idx = idx
        self.mass = mass
        self.mono_mz = mono_mz
        self.charge = charge
        self.intensity = intensity
        self.mz_low = mz_low
        self.mz_high = mz_high
        self.rt_low = rt_low
        self.rt_high = rt_high
        self.rt_apex = rt_apex
        self.ecscore = ecscore
        self.xic = xic
        self.envelope = envelope


class MultiChargeFeature:
    def __init__(self, feature_id, min_scan, max_scan, min_charge, max_charge, mass,
                 rep_charge, rep_mz, abundance, min_elution_time, max_elution_time,
                 apex_elution_time, elution_length):
        self.feature_id = feature_id
        self.min_scan = min_scan
        self.max_scan = max_scan
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.mass = mass
        self.rep_charge = rep_charge
        self.rep_mz = rep_mz
        self.abundance = abundance
        self.min_elution_time = min_elution_time
        self.max_elution_time = max_elution_time
        self.apex_elution_time = apex_elution_time
        self.elution_length = elution_length
