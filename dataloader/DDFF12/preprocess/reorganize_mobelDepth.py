import os
import shutil
import numpy as np

ROOT = 'YOUR_DOWNLOAD_FOLDER' # contain both aligned and src if download from https://www.supasorn.com/dffdownload.html
ALIGNED_PTH = ROOT + '/Aligned/' # path to  "Photos, Calbration, Results: Download" files
SRC_PTH = ROOT + '/depth_from_focus_data2/calibration/' # path to "Aligned focus stack: Download" files

DUMP_PTH = 'YOUR_DUMP_PTH' # Place to save

map_dict = {'metals': 'metal', 'largemotion':'GTLarge', 'smallmotion':'GTSmall', 'zeromotion': 'GT'}
img_folders = os.listdir(SRC_PTH)
print(img_folders)
for root, dir, files in os.walk(ALIGNED_PTH):
    for d in dir:
        if d in img_folders or d in map_dict:
            if d in map_dict:
                _d = map_dict[d]
            else:
                _d = d
            shutil.copytree('{}/{}'.format(root, d), DUMP_PTH + d)

            calib_f = np.genfromtxt('{}/{}/calibrated.txt'.format(SRC_PTH, _d), usecols=0)
            foc_dpth = calib_f[:-1]
            np.savetxt( '{}/{}/focus_dpth.txt'.format(DUMP_PTH, d),np.round(foc_dpth, 6), fmt='%5.6f')

