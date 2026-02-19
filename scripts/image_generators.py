import numpy as np
import hashlib
from typing import Tuple, Dict, Any

from config import IMAGE_RESOLUTIONS, FIBROSIS_STAGES, RANDOM_SEED
from utils import get_patient_id  # though not directly used here, may be for consistency


# -------------------- Helper: Deterministic Patient Anatomy --------------------
def _get_patient_anatomy_params(patient_id: str, modality: str) -> Dict[str, Any]:
    """
    Generate deterministic anatomical parameters for a patient, consistent across modalities.

    Args:
        patient_id: Patient identifier string.
        modality: Modality name (used to seed differently if needed, but anatomy should be same).

    Returns:
        Dictionary with keys: 'liver_center', 'liver_axes', 'rotation_angle', 'liver_intensity_base'.
    """
    # Create a seed from patient_id (hash to integer)
    seed_str = f"{patient_id}_anatomy"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.RandomState(seed)

    # Resolution from config (use MRI for anatomy, but shape independent)
    res = IMAGE_RESOLUTIONS["MRI"]  # Use MRI resolution for anatomical dimensions
    h, w = res

    # Liver region: an ellipse with random center near image center, random axes
    center_x = int(w * (0.4 + 0.2 * rng.rand()))  # between 0.4w and 0.6w
    center_y = int(h * (0.4 + 0.2 * rng.rand()))  # between 0.4h and 0.6h
    liver_center = (center_y, center_x)  # (row, col)

    # Axes lengths (semi-major and semi-minor) as fractions of image size
    axis_major = int(min(h, w) * (0.2 + 0.1 * rng.rand()))  # 20-30% of smaller dimension
    axis_minor = int(axis_major * (0.6 + 0.3 * rng.rand()))  # 60-90% of major
    liver_axes = (axis_major, axis_minor)

    # Rotation angle in degrees
    rotation_angle = rng.uniform(-30, 30)

    # Base intensity (used as mean for liver region) - will be modality-specific later
    # But we keep it here for potential cross-modality consistency
    liver_intensity_base = rng.uniform(0.5, 0.8)  # normalized intensity

    return {
        "center": liver_center,
        "axes": liver_axes,
        "angle": rotation_angle,
        "intensity_base": liver_intensity_base,
        "rng": rng  # Return the rng for modality-specific randomness (optional)
    }


# -------------------- Helper: Create Elliptical Mask --------------------
def _create_elliptical_mask(shape: Tuple[int, int], center: Tuple[int, int], axes: Tuple[int, int],
                            angle: float) -> np.ndarray:
    """
    Create a binary mask of an ellipse.

    Args:
        shape: Image shape (height, width).
        center: (row, col) of ellipse center.
        axes: (semi-major, semi-minor) lengths.
        angle: Rotation angle in degrees.

    Returns:
        2D boolean mask.
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    y_centered = y - center[0]
    x_centered = x - center[1]

    # Rotate coordinates
    theta = np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = x_centered * cos_t - y_centered * sin_t
    y_rot = x_centered * sin_t + y_centered * cos_t

    # Ellipse equation
    mask = (x_rot ** 2 / axes[1] ** 2 + y_rot ** 2 / axes[0] ** 2) <= 1
    return mask


# -------------------- Modality Generators --------------------
def generate_mri_image(patient_id: str, fibrosis_stage: str, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Generate a synthetic MRI image of the liver.

    Args:
        patient_id: Patient identifier.
        fibrosis_stage: One of F0-F4.
        resolution: (height, width) of the output image.

    Returns:
        2D float32 array with values in [0, 1].
    """
    h, w = resolution
    params = _get_patient_anatomy_params(patient_id, "MRI")
    rng = params["rng"]  # Use same RNG for consistency

    # Create mask
    mask = _create_elliptical_mask((h, w), params["center"], params["axes"], params["angle"])

    # Background: low intensity with slight noise
    background = rng.normal(loc=0.1, scale=0.02, size=(h, w)).clip(0, 1)

    # Liver region: base intensity modulated by fibrosis stage
    stage_index = FIBROSIS_STAGES.index(fibrosis_stage)

    # Base liver intensity increases with stage (more fibrotic = brighter on T2)
    liver_base = 0.5 + 0.1 * stage_index  # 0.5 to 0.9
    liver_intensity = rng.normal(loc=liver_base, scale=0.05, size=(h, w))

    # Add texture: for higher stages, add random bright spots (nodules)
    if stage_index >= 2:  # F2 and above
        num_spots = rng.poisson(lam=5 + 3 * (stage_index - 2))
        for _ in range(num_spots):
            spot_y = rng.randint(0, h)
            spot_x = rng.randint(0, w)
            if mask[spot_y, spot_x]:
                # Add a small bright disk
                rr, cc = np.ogrid[:h, :w]
                dist = np.sqrt((rr - spot_y) ** 2 + (cc - spot_x) ** 2)
                spot_mask = dist <= rng.randint(3, 8)
                liver_intensity[spot_mask] += rng.uniform(0.2, 0.4)

    # Combine: where mask is True, use liver_intensity; else background
    image = np.where(mask, liver_intensity, background)

    # Ensure [0,1] range
    image = np.clip(image, 0, 1).astype(np.float32)
    return image


def generate_ct_image(patient_id: str, fibrosis_stage: str, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Generate a synthetic CT image of the liver (simulated Hounsfield units normalized to [0,1]).

    Args:
        patient_id: Patient identifier.
        fibrosis_stage: One of F0-F4.
        resolution: (height, width) of the output image.

    Returns:
        2D float32 array with values in [0, 1].
    """
    h, w = resolution
    params = _get_patient_anatomy_params(patient_id, "CT")
    rng = params["rng"]

    mask = _create_elliptical_mask((h, w), params["center"], params["axes"], params["angle"])

    # Background (air) low, some noise
    background = rng.normal(loc=0.05, scale=0.01, size=(h, w)).clip(0, 1)

    # Liver CT: typical HU ~50-60, but we normalize to [0,1] where 0~ -1000, 1~ +1000.
    # We'll map: 0 -> -1000, 1 -> +1000. So liver around 0.5 corresponds to 0 HU? Let's just use [0,1] directly.
    stage_index = FIBROSIS_STAGES.index(fibrosis_stage)

    # Liver mean intensity increases slightly with fibrosis (more dense)
    liver_mean = 0.5 + 0.05 * stage_index
    liver_intensity = rng.normal(loc=liver_mean, scale=0.02, size=(h, w))

    # For advanced stages, add some high-density spots (calcifications)
    if stage_index >= 3:  # F3, F4
        num_spots = rng.poisson(lam=3)
        for _ in range(num_spots):
            spot_y = rng.randint(0, h)
            spot_x = rng.randint(0, w)
            if mask[spot_y, spot_x]:
                rr, cc = np.ogrid[:h, :w]
                dist = np.sqrt((rr - spot_y) ** 2 + (cc - spot_x) ** 2)
                spot_mask = dist <= rng.randint(2, 5)
                liver_intensity[spot_mask] += rng.uniform(0.3, 0.6)

    image = np.where(mask, liver_intensity, background)
    image = np.clip(image, 0, 1).astype(np.float32)
    return image


def generate_ultrasound_image(patient_id: str, fibrosis_stage: str, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Generate a synthetic ultrasound image of the liver with speckle noise.

    Args:
        patient_id: Patient identifier.
        fibrosis_stage: One of F0-F4.
        resolution: (height, width) of the output image.

    Returns:
        2D float32 array with values in [0, 1].
    """
    h, w = resolution
    params = _get_patient_anatomy_params(patient_id, "Ultrasound")
    rng = params["rng"]

    mask = _create_elliptical_mask((h, w), params["center"], params["axes"], params["angle"])

    # Background (anechoic) low
    background = rng.gamma(shape=1, scale=0.01, size=(h, w)).clip(0, 1)

    # Liver parenchyma: speckle noise modelled as Gamma or Rayleigh.
    # For simplicity, use Gamma with shape and scale.
    stage_index = FIBROSIS_STAGES.index(fibrosis_stage)

    # More fibrosis -> higher echogenicity (brighter) and coarser texture (higher shape parameter)
    shape = 2.0 + 1.0 * stage_index  # 2 to 6
    scale = 0.1 + 0.02 * stage_index  # scale increases brightness
    liver_intensity = rng.gamma(shape=shape, scale=scale, size=(h, w))

    # Add some bright reflections for advanced stages
    if stage_index >= 2:
        num_spots = rng.poisson(lam=8)
        for _ in range(num_spots):
            spot_y = rng.randint(0, h)
            spot_x = rng.randint(0, w)
            if mask[spot_y, spot_x]:
                rr, cc = np.ogrid[:h, :w]
                dist = np.sqrt((rr - spot_y) ** 2 + (cc - spot_x) ** 2)
                spot_mask = dist <= rng.randint(2, 4)
                liver_intensity[spot_mask] += rng.uniform(0.5, 1.0)

    image = np.where(mask, liver_intensity, background)
    # Normalize to [0,1] by clipping
    image = np.clip(image, 0, 1).astype(np.float32)
    return image