import sys

import numpy as np

import maccel

from utils import preprocess_resnet50, IMAGENET_LABEL

RESNET50_MXQ_PATH = 'resnet50.mxq'
DATASET = [
    ('ILSVRC2012_val_00000001.JPEG', 65),
    ('imagenet_sample/img/ILSVRC2012_val_00023986.JPEG', 4),
    ('imagenet_sample/img/ILSVRC2012_val_00044624.JPEG', 9),
]

def main(args):
    mxq_path = RESNET50_MXQ_PATH
    if len(args) == 2:
        mxq_path = args[1]

    acc = maccel.Accelerator()
    model = maccel.Model(mxq_path)
    model.launch(acc)
    for filename, label in DATASET:
        img = preprocess_resnet50(filename)
        result = model.infer([img])[0]
        predict = np.array(result).argmax()

        print(f'label   : ({label}, {IMAGENET_LABEL[label]})')
        print(f'predict : ({predict}, {IMAGENET_LABEL[predict]})')

    model.dispose()


if __name__ == '__main__':
    main(sys.argv)
