from qubee.calibration import make_calib_man
from utils import preprocess_resnet50

make_calib_man(
    pre_ftn=preprocess_resnet50,
    data_dir="./Calibration_Images/",
    save_dir="./Calibaration_Images_npy/",
    save_name="classification_calibrationDataset",
    max_size=1024
    )



