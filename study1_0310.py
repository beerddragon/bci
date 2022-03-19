import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# #### 1) Loading data ####
# EEG and MEG data from one subject performing an audiovisual experiment + structural MRI scans
sample_data_folder = mne.datasets.sample.data_path()

# unfiltered data
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

print(raw)
print(raw.info)

# some examples of raw.info:
print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
print(raw.info['sfreq'], 'Hz')            # sampling frequency
print(raw.info['description'], '\n')      # miscellaneous acquisition info

# raw 객체의 built-in plotting 메서드
raw.plot_psd(fmax=50)
raw.plot(duration=5, n_channels=30)

# #### 2) Preprocessing ####

# filtering & resampling data
# use just 60 seconds of data and mag channels, to save memory
raw.crop(0, 60).pick_types(meg='mag', stim=True).load_data()

# looking for slow drift in the data
raw.plot(duration=60, proj=False, n_channels=len(raw.ch_names),
         remove_dc=False)

# for cutoff frequency
for cutoff in (0.1, 0.2):
    raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
    fig = raw_highpass.plot(duration=60, proj=False,
                            n_channels=len(raw.ch_names), remove_dc=False)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
                 weight='bold')

# filter visualization
filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'],
                                         l_freq=0.2, h_freq=None)
mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.01, 5))

# finding power line noise
# adding arrows at artifacts
def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)

# applying notch filter to raw object
meg_picks = mne.pick_types(raw.info, meg=True)
freqs = (60, 120, 180, 240)
raw_notch = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)

for title, data in zip(['Un', 'Notch '], [raw, raw_notch]):
    fig = data.plot_psd(fmax=250, average=True)
    fig.subplots_adjust(top=0.85)
    fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
    add_arrows(fig.axes[:2])

# notch filtering with spectrum fitting
raw_notch_fit = raw.copy().notch_filter(
    freqs=freqs, picks=meg_picks, method='spectrum_fit', filter_length='10s')

for title, data in zip(['Un', 'spectrum_fit '], [raw, raw_notch_fit]):
    fig = data.plot_psd(fmax=250, average=True)
    fig.subplots_adjust(top=0.85)
    fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
    add_arrows(fig.axes[:2])

# resampling
raw_downsampled = raw.copy().resample(sfreq=200)

for data, title in zip([raw, raw_downsampled], ['Original', 'Downsampled']):
    fig = data.plot_psd(average=True)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title)
    plt.setp(fig.axes, xlim=(0, 300))

# ECG detection
# unfiltered data
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(0, 60).load_data()

#  extract epochs centered around the detected heartbeat artifacts
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
ecg_epochs.plot_image(combine='mean')

# image plot without drift
# ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, baseline=(-0.5, -0.2))
# ecg_epochs.plot_image(combine='mean')

# baseline correction
avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))

# visualization of spatial pattern of the associated field
avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))

# combined scalp field maps and ERP/F plot
avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])

# EOG detection
# find artifacts & extract epochs
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))

# visualization
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()

# Bad channel marking
from copy import deepcopy

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)

print(raw.info['bads'])

# look at neighboring channels (EEG 050-EEG 059)
picks = mne.pick_channels_regexp(raw.ch_names, regexp='EEG 05.')
raw.plot(order=picks, n_channels=len(picks))

# look at neighboring channels (MEG 2__3)
picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')
raw.plot(order=picks, n_channels=len(picks))

# editing bad channel lists
original_bads = deepcopy(raw.info['bads'])
raw.info['bads'].append('EEG 050')               # add a single channel
raw.info['bads'].extend(['EEG 051', 'EEG 052'])  # add a list of channels
bad_chan = raw.info['bads'].pop(-1)  # remove the last entry in the list
raw.info['bads'] = original_bads     # change the whole list at once

# ERP/F plots of raw data
raw2 = raw.copy()
raw2.info['bads'] = []
events = mne.find_events(raw2, stim_channel='STI 014')
epochs = mne.Epochs(raw2, events=events)['2'].average().plot()

