#!/usr/bin/python
import magphase as mp
from model import SampleRNN

mp.analysis_for_acoustic_modelling(path.join(wavdir, sentence+'.wav'),
                                   outdir,
                                   mag_dim=cfg_data.get('nm', 60),
                                   phase_dim=cfg_data.get('np', 45),
                                   b_const_rate=True)
