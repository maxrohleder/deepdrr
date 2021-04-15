#! python3

import logging
import os

import numpy as np
from tifffile import imsave
from pathlib import Path
from time import time
import xmltodict

from deepdrr import Volume, Projector
from deepdrr import geo

logger = logging.getLogger(__name__)


def spin_matrices_from_xml(path_to_projection_matrices: str) -> list[np.ndarray]:
    with open(path_to_projection_matrices) as fd:
        contents = xmltodict.parse(fd.read())
        matrices = contents['hdr']['ElementList']['PROJECTION_MATRICES']
        # backprojection matrices project 2d-projections into 3d-space (homogenous coordinates allow for translation)
        proj_mat = []
        # parsing the ordered dict to numpy matrices
        for i, key in enumerate(matrices.keys()):
            value_string = matrices[key]
            # expecting 12 entries to construct 3x4 matrix in row-major order (C-contiguous)
            proj_mat.append(np.array(value_string.split(" "), order='C').astype(np.float32).reshape((3, 4)))
    logger.info(f"loaded {len(proj_mat)} projection matrices")
    return proj_mat


def main():
    # Paths
    ct_volume = Path('Knee01.dcm')

    # load volume from dicom
    volume = Volume.from_dicom(ct_volume)

    # camera intrinsics (only used to determine the sensor_size?) TODO maybe make this more explicit
    camera_intrinsics = geo.CameraIntrinsicTransform.from_sizes(
        sensor_size=(976, 976),
        pixel_size=0.313,
        source_to_detector_distance=1164,
    )

    # load projection matrices as list of np.ndarrays
    projection_matrices = spin_matrices_from_xml('SpinProjMatrix_5proj.xml')

    t = time()
    with Projector(
            volume=volume,
            camera_intrinsics=camera_intrinsics,
            step=0.1,  # stepsize along projection ray, measured in voxels
            mode='linear',
            max_block_index=200,
            spectrum='90KV_AL40',
            photon_count=100000,
            add_scatter=False,
            threads=8,
            neglog=True,
    ) as projector:
        logger.info(f"starting projection after {time() - t} seconds")
        images = projector.project_with_matrices(*projection_matrices)
    dt = time() - t
    logger.info(f"projected {images.shape[0]} views in {dt:.03f}s")

    # create output folders
    out_path = os.path.abspath(r"../../generated_data")
    os.makedirs(out_path, exist_ok=True)  # directories might already exist

    # save drr in tiff (float32)
    out = os.path.abspath(os.path.join(out_path, "new_drr.tiff"))
    imsave(out, images)
    print("saved a tiff file to ", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
