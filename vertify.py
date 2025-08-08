import os, numpy as np
def check_split(split='val'):
    base = os.path.join('dataset', split)
    img_dir, tgt_dir = os.path.join(base,'images'), os.path.join(base,'targets')
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])[:100]
    pos=tot=0
    for f in files:
        pid = f.split('_')[0]
        bev = np.load(os.path.join(tgt_dir, f"{pid}_bev.npy"))
        pos += (bev>0).sum(); tot += bev.size
    print(split, 'positive % =', 100*pos/tot, '%')
check_split('train'); check_split('val')