#! python3

import logging
import os

import numpy as np
from tifffile import imsave
from pathlib import Path
from time import time
import xmltodict

from deepdrr import Volume, CArm, Projector
from deepdrr import geo

logger = logging.getLogger(__name__)


def main():
    # Paths
    ct_volume = Path('Knee01.dcm')

    # load volume from dicom
    volume = Volume.from_dicom(ct_volume)

    # defines the C-Arm device, which is a convenience class for positioning the Camera.
    carm = CArm(
        isocenter=geo.point(0, 0, 0),
        isocenter_distance=800,
    )

    # camera intrinsics (only used to determine the sensor_size?) TODO maybe make this more explicit
    camera_intrinsics = geo.CameraIntrinsicTransform.from_sizes(
        sensor_size=(976, 976),
        pixel_size=0.313,
        source_to_detector_distance=1164,
    )

    # Angles to take projections over
    min_theta = 90
    max_theta = 91.5
    min_phi = 0
    max_phi = 201
    spacing_theta = 1
    spacing_phi = 40

    t = time()
    with Projector(
            volume=volume,
            camera_intrinsics=camera_intrinsics,
            carm=carm,
            step=0.1,  # stepsize along projection ray, measured in voxels
            mode='linear',
            max_block_index=200,
            spectrum='90KV_AL40',
            photon_count=100000,
            add_scatter=False,
            threads=8,
            neglog=True,
    ) as projector:
        images = projector.project_over_carm_range(
            (min_phi, max_phi, spacing_phi),
            (min_theta, max_theta, spacing_theta))
    dt = time() - t
    logger.info(f"projected {images.shape[0]} views in {dt:.03f}s")

    # create output folders
    out_path = os.path.abspath(r"../../generated_data")
    os.makedirs(out_path, exist_ok=True)  # directories might already exist

    # save drr in tiff (float32)
    out = os.path.abspath(os.path.join(out_path, "example_trajectory.tiff"))
    imsave(out, images)
    print("saved a tiff file to ", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
