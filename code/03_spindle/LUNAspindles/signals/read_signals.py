# -*- coding: utf-8 -*-
"""
LUNAspindles.signals.read_signals
---------------------------------
This module loads biosignals from EDF+(C/D) files and MAT files
EDF+ reader was dapted from 2012 Boris Reuderink
(c) 2019 by Noor Adra (CDAC Westover Lab, MGH)
license?
"""

import re, datetime, operator, logging, functools
from collections import namedtuple
import scipy.io as sio
import numpy as np
import h5py

EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)

class EDFEndOfData: pass

def tal(tal_str):
  """Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
  stream.
  """
  exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
    '(?:\x14(?P<duration>\d+(?:\.\d*)))?' + \
    '(\x14(?P<annotation>[^\x00]*))?' + \
    '(?:\x14\x00)'

  def annotation_to_list(annotation):
    return annotation.split('\x14') if annotation else []

  def parse(dic):
    return (
      float(dic['onset']), 
      float(dic['duration']) if dic['duration'] else 0.,
      annotation_to_list(dic['annotation']))

  return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]

def edf_header(f):
  h = {}
  assert f.tell() == 0  # check file position
  assert f.read(8).decode("utf-8") == '0       '

  # recording info)
  h['local_subject_id'] = f.read(80).strip().decode("utf-8")
  h['local_recording_id'] = f.read(80).strip().decode("utf-8")

  # parse timestamp
  (day, month, year) = [int(x) for x in re.findall('(\d+)',
                                                   f.read(8).decode("utf-8"))]
  (hour, minute, sec)= [int(x) for x in re.findall('(\d+)',
                                                   f.read(8).decode("utf-8"))]
  h['date_time'] = str(datetime.datetime(year + 2000, month,
                                         day, hour, minute, sec))

  # misc
  header_nbytes = int(f.read(8))
  subtype = f.read(44)[:5].decode("utf-8")
  h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
  h['contiguous'] = subtype != 'EDF+D'
  h['n_records'] = int(f.read(8))
  h['record_length'] = float(f.read(8).decode("utf-8"))  # in seconds
  nchannels = h['n_channels'] = int(f.read(4).decode("utf-8"))

  # read channel info
  channels = range(h['n_channels'])
  h['label'] = [f.read(16).strip().decode("utf-8") for n in channels]
  h['transducer_type'] = [f.read(80).strip().decode("utf-8") for n in channels]
  h['units'] = [f.read(8).strip().decode("utf-8") for n in channels]
  h['physical_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['physical_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['prefiltering'] = [f.read(80).strip().decode("utf-8")  for n in channels]
  h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
  f.read(32 * nchannels).decode("utf-8")   # reserved
  
  assert f.tell() == header_nbytes
  return h 

class BaseEDFReader:
  def __init__(self, file):
    self.file = file

  def read_header(self):
    self.header = h = edf_header(self.file)

    # calculate ranges for rescaling
    self.dig_min = h['digital_min']
    self.phys_min = h['physical_min']
    phys_range = h['physical_max'] - h['physical_min']
    dig_range = h['digital_max'] - h['digital_min']
    self.gain = phys_range / dig_range

  def read_raw_record(self):
    """Read a record with data and return a list containing arrays with raw
    bytes.
    """
    result = []
    for nsamp in self.header['n_samples_per_record']:
        samples = self.file.read(nsamp * 2)
        if len(samples) != nsamp * 2:
            raise EDFEndOfData
        result.append(samples)
    return result

  def convert_record(self, raw_record):
    """Convert a raw record to a (time, signals, events) tuple based on
    information in the header.
    """
    h = self.header
    dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
    time = float('nan')
    signals = []
    events = []
    for (i, samples) in enumerate(raw_record):
        if h['label'][i] == EVENT_CHANNEL:
            ann = tal(samples.decode())
            time = ann[0][0]
            events.extend(ann[1:])
        else:
                # 2-byte little-endian integers
            dig = np.frombuffer(samples, '<i2').astype(np.float32)
            phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
            signals.append(phys)
    return time, signals, events

  def read_record(self):
    return self.convert_record(self.read_raw_record())

  def records(self):
    """
    Record generator.
    """
    try:
      while True:
        yield self.read_record()
    except: 
      pass



def load_edf(edffile):
  """
  Loads an EDF+ file.

  Very basic reader for EDF and EDF+ files. While BaseEDFReader does support
  exotic features like non-homogeneous sample rates and loading only parts of
  the stream, load_edf expects a single fixed sample rate for all channels and
  tries to load the whole file.

  Parameters
  ----------
  edffile : file-like object or string

  Returns
  -------
  Named tuple with the fields:
    X : NumPy array with shape p by n.
      Raw recording of n samples in p dimensions.
    sample_rate : float
      The sample rate of the recording. Note that mixed sample-rates are not
      supported.
    chan_lab : list of length p with strings
      The labels of the sensors used to record X.
    channel_info : list of dictionaries with info for each channel:
                  -label
                  -dimension
                  -sample_rate
                  -physical_max
                  -physical_min
                  -digital_max
                  -digital_min
                  -transducer
                  -prefilter
                    
  Notes
  -----
  Code below was adjusted by Noor for Westover lab EDF+D filter 

  """
  
  if isinstance(edffile, str):
    with open(edffile, 'rb') as f:
      return load_edf(f)  # convert filename to file

  reader = BaseEDFReader(edffile)
  reader.read_header()
  h = reader.header
  log.debug('EDF header: %s' % h)
  
  # get sample rate info
  nsamp = np.unique(
    [n for (l, n) in zip(h['label'], h['n_samples_per_record'])
    if l != EVENT_CHANNEL])
  assert nsamp.size == 1, 'Multiple sample rates not supported!'
  sample_rate = float(nsamp[0]) / h['record_length']

  rectime, X, annotations = zip(*reader.records())
  X = np.hstack(X)

  chan_lab = [lab for lab in reader.header['label'] if lab != EVENT_CHANNEL]

  #channel_info is used to generate edf+ files later
  channel_info = [
                {'label': chan_lab[i],
                 'dimension': h['units'][i], #'uV',
                 'sample_rate': sample_rate,
                 'physical_max': h['physical_max'][i], #32767,
                 'physical_min': h['physical_min'][i], #-32768,
                 'digital_max': h['digital_max'][i], #32767,
                 'digital_min': h['digital_min'][i], #-32768,
                 'transducer': h['transducer_type'][i], #'E',
                 'prefilter': h['prefiltering'][i]} #''
            for i in range(len(chan_lab))]
  
  tup = namedtuple('EDF', 'X sample_rate chan_lab channel_info')
  tupleData = tup(X, sample_rate, chan_lab, channel_info)

  return tupleData
                

def load_mat(input_path):
    """
    Loads a signal.MAT file.

    Very basic reader for MAT files. Expects a single fixed sample rate for all
    channels and tries to load the whole file.

    Parameters
    ----------
    edffile : file with .mat extension

    Returns
    -------
    Named tuple with the fields:
      X : NumPy array with shape p by n.
        Raw recording of n samples in p dimensions.
      sample_rate : float
        The sample rate of the recording. Note that mixed sample-rates are not
        supported.
      chan_lab : list of length p with strings
        The labels of the sensors used to record X.
      channel_info : list of dictionaries with info for each channel:
                    -label
                    -dimension
                    -sample_rate
                    -physical_max
                    -physical_min
                    -digital_max
                    -digital_min
                    -transducer
                    -prefilter
                      
    Notes
    -----
    
    """
    # load data
    try:
        ff = sio.loadmat(input_path)
        chan_lab = [ff['hdr'][0,i]['signal_labels'][0].upper() for i in range(ff['hdr'].shape[1])]
        #print('scio')
    except:
        ff = h5py.File(input_path, 'r')
        chan_lab = [ff['hdr'][0,i]['signal_labels'][0][0].upper() for i in range(ff['hdr'].shape[1])]
        #print('h5py')
    if 's' not in ff: 
        raise Exception('No signal found.')
    
    X = ff['s']

    if 'Fs' in ff:
      sample_rate = ff['Fs']
    elif 'sfreq' in ff:
      sample_rate = ff['sfreq']
    else:
      sample_rate = 200

    #indices for channel labels
    channel_ids = [chan_lab.index(ch) for ch in chan_lab]
    channel_info = []

    to_del_i = []
    to_del_ch = []
    
    for i in channel_ids:
        try:
          channel_info.append({'label': chan_lab[i],
                    'dimension': ff['hdr'][0,i]['physical_dimension'][0],
                    'sample_rate': int(sample_rate),
                    'physical_max': int(ff['hdr'][0,i]['physical_max'][0,0]),
                    'physical_min': int(ff['hdr'][0,i]['physical_min'][0,0]),
                    'digital_max': int(ff['hdr'][0,i]['digital_max'][0,0]),
                    'digital_min': int(ff['hdr'][0,i]['digital_min'][0,0]),
                    'transducer': 'E',
                    'prefilter': ff['hdr'][0,i]['prefiltering'][0]
                               if len(ff['hdr'][0,i]['prefiltering'])>0 else ''})
        except:
            print('error')
            print(chan_lab[i])
            print('************************')
            to_del_ch.append(chan_lab[i])
            to_del_i.append(i)


    chan_lab = [ch for ch in chan_lab if ch not in to_del_ch]
    X = np.delete(X, [to_del_i],0) 

    tup = namedtuple('EDF', 'X sample_rate chan_lab channel_info')
    tupleData = tup(X, sample_rate, chan_lab, channel_info) 

    return tupleData
