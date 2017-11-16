from argparse import ArgumentParser
from os.path import abspath, expanduser

from utils import core as util
from utils.Logger import Logger


log = Logger(lvl='debug', delay=45)
parser = ArgumentParser("ImageCaptionLearn_py: Feature File Splitter")
parser.add_argument("--feats_file", type=str, help="The feature file to split into " +
                    "intra-caption and inter-caption files")
parser.add_argument('--ordered_intra', action='store_true',
        help='Whether to split intra-caption file into ordered ij and ji files')
args = parser.parse_args()
arg_dict = vars(args)
util.dump_args(arg_dict, log)
feats_file = abspath(expanduser(arg_dict['feats_file']))

intra_cap_ij = list()
intra_cap_ji = list()
inter_cap = list()
with open(feats_file, 'r') as f:
    for line in f:
        line_parts = line.split(" # ")
        id_parts = line_parts[1].split(";")
        cap_1 = None
        cap_2 = None
        m_1 = None
        m_2 = None
        for id_part in id_parts:
            kv_pair = id_part.split(":")
            if kv_pair[0] == 'caption_1':
                cap_1 = int(kv_pair[1])
            elif kv_pair[0] == 'caption_2':
                cap_2 = int(kv_pair[1])
            elif kv_pair[0] == 'mention_1':
                m_1 = int(kv_pair[1])
            elif kv_pair[0] == 'mention_2':
                m_2 = int(kv_pair[1])
        #endfor
        if cap_1 is not None and cap_2 is not None and cap_1 == cap_2:
            if m_1 < m_2:
                intra_cap_ij.append(line)
            else:
                intra_cap_ji.append(line)
        else:
            inter_cap.append(line)
        #endif
    #endfor
#endwith

if arg_dict['ordered_intra']:
    with open(feats_file.replace(".feats", "_intra_ij.feats"), 'w') as f:
        for line in intra_cap_ij:
            f.write(line)
    with open(feats_file.replace(".feats", "_intra_ji.feats"), 'w') as f:
        for line in intra_cap_ji:
            f.write(line)
else:
    with open(feats_file.replace(".feats", "_intra.feats"), 'w') as f:
        for line in intra_cap_ij:
            f.write(line)
        for line in intra_cap_ji:
            f.write(line)
#endif

with open(feats_file.replace(".feats", "_inter.feats"), 'w') as f:
    for line in inter_cap:
        f.write(line)
