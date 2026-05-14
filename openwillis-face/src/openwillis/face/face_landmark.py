# author:    Vijay Yadav
# website:   http://www.bklynhlth.com

# import the required packages
from collections import defaultdict
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import json
import os
import logging
import mediapipe as mp

from .util import crop_with_padding_and_center, get_speaking_probabilities, split_speaking_df, get_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BBoxDict = Dict[str, Any]
BBoxList = List[BBoxDict]
RotationMatrixList = List[np.ndarray]
TimingStats = Dict[str, Dict[str, float]]

NUM_LANDMARKS = 468
FRAME_META_COLUMNS = ["frame", "time"]
LANDMARK_COORD_COLUMNS = [
    f"lmk{index:03d}_{axis}"
    for axis in ("x", "y", "z")
    for index in range(1, NUM_LANDMARKS + 1)
]
LANDMARK_DISPLACEMENT_COLUMNS = [
    f"lmk{index:03d}" for index in range(1, NUM_LANDMARKS + 1)
]
EMPTY_LANDMARK_ROW = np.full(len(LANDMARK_COORD_COLUMNS), np.nan, dtype=float)
TIMING_ENABLED = os.getenv("OPENWILLIS_FACE_TIMING", "").lower() in {"1", "true", "yes", "on"}

_TIMING_STATS: TimingStats = defaultdict(lambda: {"count": 0, "total_seconds": 0.0})
_LANDMARK_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def _current_timestamp() -> str:
    """
    Return an ISO-8601 timestamp with millisecond precision for timing logs.
    """
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _reset_timing_stats() -> None:
    """
    Clear accumulated timing statistics for a new facial_expressivity run.
    """
    _TIMING_STATS.clear()


def _normalize_cache_value(value: Any) -> Any:
    """
    Convert bbox data into a stable JSON-serializable representation.
    """
    if isinstance(value, dict):
        return {key: _normalize_cache_value(val) for key, val in sorted(value.items())}

    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and np.isnan(value):
        return "NaN"

    return value


def _get_landmark_cache_key(path: str, bbox_list: BBoxList) -> Tuple[str, str]:
    """
    Create a stable cache key for raw landmarks from the video path and bbox list.
    """
    normalized_path = os.path.abspath(path)
    bbox_fingerprint = json.dumps(
        _normalize_cache_value(bbox_list),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return normalized_path, bbox_fingerprint


def _get_empty_landmark_dataframe(frame: int, fps: int) -> pd.DataFrame:
    """
    Build a one-row landmark dataframe filled with NaNs for undetected frames.
    """
    time_value = frame / fps if fps else np.nan
    df_meta = pd.DataFrame([[frame, time_value]], columns=FRAME_META_COLUMNS)
    df_coord = pd.DataFrame([EMPTY_LANDMARK_ROW.copy()], columns=LANDMARK_COORD_COLUMNS)
    return pd.concat([df_meta, df_coord], axis=1)


def _log_timing_summary() -> None:
    """
    Log a timing summary sorted by total elapsed time.
    """
    if not TIMING_ENABLED or not _TIMING_STATS:
        return

    logger.info("facial_expressivity timing summary start")
    summary_rows = sorted(
        _TIMING_STATS.items(),
        key=lambda item: item[1]["total_seconds"],
        reverse=True,
    )

    for function_name, stats in summary_rows:
        count = int(stats["count"])
        total_seconds = stats["total_seconds"]
        average_seconds = total_seconds / count if count else 0.0
        logger.info(
            "timing_summary function=%s calls=%s total_seconds=%.3f avg_seconds=%.3f",
            function_name,
            count,
            total_seconds,
            average_seconds,
        )

    logger.info("facial_expressivity timing summary end")


def _timed(func):
    """
    Decorator that logs start and end timestamps and records cumulative timing stats.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TIMING_ENABLED:
            return func(*args, **kwargs)

        function_name = func.__name__
        start_timestamp = _current_timestamp()
        start_time = perf_counter()
        logger.info("timing_start function=%s timestamp=%s", function_name, start_timestamp)

        try:
            return func(*args, **kwargs)
        finally:
            end_time = perf_counter()
            elapsed_seconds = end_time - start_time
            end_timestamp = _current_timestamp()
            _TIMING_STATS[function_name]["count"] += 1
            _TIMING_STATS[function_name]["total_seconds"] += elapsed_seconds
            logger.info(
                "timing_end function=%s timestamp=%s elapsed_seconds=%.3f",
                function_name,
                end_timestamp,
                elapsed_seconds,
            )

    return wrapper


@contextmanager
def _timed_block(block_name: str):
    """
    Context manager for timing internal stages inside a function.
    """
    if not TIMING_ENABLED:
        yield
        return

    start_timestamp = _current_timestamp()
    start_time = perf_counter()
    logger.info("timing_start function=%s timestamp=%s", block_name, start_timestamp)

    try:
        yield
    finally:
        elapsed_seconds = perf_counter() - start_time
        end_timestamp = _current_timestamp()
        _TIMING_STATS[block_name]["count"] += 1
        _TIMING_STATS[block_name]["total_seconds"] += elapsed_seconds
        logger.info(
            "timing_end function=%s timestamp=%s elapsed_seconds=%.3f",
            block_name,
            end_timestamp,
            elapsed_seconds,
        )


@_timed
def get_config(filepath: str, json_file: str) -> Dict[str, Any]:
    """
    ------------------------------------------------------------------------------------------------------

    This function reads the configuration file containing the column names for the output dataframes,
    and returns the contents of the file as a dictionary.

    Parameters:
    ...........
    filepath : str
        The path to the configuration file.
    json_file : str
        The name of the configuration file.

    Returns:
    ...........
    measures: A dictionary containing the names of the columns in the output dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    dir_name = os.path.dirname(filepath)
    measure_path = os.path.abspath(os.path.join(dir_name, f"config/{json_file}"))

    file = open(measure_path)
    measures = json.load(file)
    return measures


@_timed
def init_facemesh() -> Any:
    """
    ---------------------------------------------------------------------------------------------------

    This function initializes a Facemesh object from the Mediapipe library, with a minimum detection
    confidence of 0.5. It returns the Facemesh object.

    Parameters:
    ............
    None

    Returns:
    ............
    face_mesh : Mediapipe object
        Facemesh object with minimum detection confidence of 0.5

    ---------------------------------------------------------------------------------------------------
    """

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    return face_mesh

@_timed
def filter_coord(result: Any) -> np.ndarray:
    """
    ---------------------------------------------------------------------------------------------------

    This function takes the output from a Facemesh object and returns the flattened 3D coordinates for
    each facial landmark.

    Parameters:
    ............
    result : Mediapipe object
        Output from a Facemesh object

    Returns:
    ............
    coord_row : numpy.ndarray
        Flattened landmark coordinates ordered as all x values, then y values, then z values

    ---------------------------------------------------------------------------------------------------
    """

    if not result.multi_face_landmarks:
        return EMPTY_LANDMARK_ROW.copy()

    landmarks = result.multi_face_landmarks[0].landmark[:NUM_LANDMARKS]
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    z_coords = [landmark.z for landmark in landmarks]

    if len(landmarks) < NUM_LANDMARKS:
        missing_count = NUM_LANDMARKS - len(landmarks)
        x_coords.extend([np.nan] * missing_count)
        y_coords.extend([np.nan] * missing_count)
        z_coords.extend([np.nan] * missing_count)

    return np.asarray(x_coords + y_coords + z_coords, dtype=float)

@_timed
def process_and_format_face_mesh(
    img: np.ndarray,
    face_mesh: Any,
    ) -> np.ndarray:
    """
    Process the given image using the face_mesh model and format the resulting face landmarks.

    Args:
        img (numpy.ndarray): The input image.
        face_mesh: The face_mesh model.
    Returns:
        numpy.ndarray: Flattened landmark coordinates for the frame.
    """
    with _timed_block("process_and_format_face_mesh.face_mesh_process"):
        result = face_mesh.process(img)

    with _timed_block("process_and_format_face_mesh.filter_coord"):
        coord_row = filter_coord(result)

    return coord_row

@_timed
def crop_and_process_face_mesh(
    img: np.ndarray,
    face_mesh: Any,
    bbox: BBoxDict,
    ) -> np.ndarray:
    """
    ---------------------------------------------------------------------------------------------------
    Crop and process the face mesh on the given image.
    .......
    Args:
        img (numpy.ndarray): The input image.
        face_mesh (object): The face mesh object.
        bbox (dict): The bounding box coordinates of the face.
    .......
    Returns:
        numpy.ndarray: Flattened landmark coordinates for the frame.
    ---------------------------------------------------------------------------------------------------
    """
    if bbox and not np.isnan(bbox['bb_x']):
        cropped_img = crop_with_padding_and_center(img, bbox)
        return process_and_format_face_mesh(cropped_img, face_mesh)

    return EMPTY_LANDMARK_ROW.copy()


@_timed
def run_facemesh(path: str, bbox_list: BBoxList = []) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file as input, runs Facemesh on the image, and returns a list
    of dataframes containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    bbox_list : list
        List of bounding boxes for each frame in the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        DataFrame containing landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    coord_rows: List[np.ndarray] = []
    meta_rows: List[Tuple[int, float]] = []
    frame = 0

    try:

        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        len_bbox_list = len(bbox_list)

        if (len_bbox_list>0) & (num_frames != len_bbox_list):
            raise ValueError('Number of frames in video and number of bounding boxes do not match')
        
        face_mesh = init_facemesh()

        while True:
            try:

                ret_type, img = cap.read()
                if ret_type is not True:
                    break
                time_value = frame / fps if fps else np.nan
                with _timed_block("run_facemesh.cvtColor"):
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if len_bbox_list==0:

                    coord_row = process_and_format_face_mesh(img_rgb, face_mesh)
                    
                else:
                    
                    bbox = bbox_list[frame]
                    coord_row = crop_and_process_face_mesh(img_rgb, face_mesh, bbox)

            except Exception as e:
                logger.debug("Face mesh failed on frame %s with error: %s", frame, e)
                coord_row = EMPTY_LANDMARK_ROW.copy()

            meta_rows.append((frame, time_value))
            coord_rows.append(coord_row)
            frame +=1

    except Exception as e:
        logger.info(f'Face not detected by mediapipe file: {path} & Error: {e}')

    finally:
        if 'cap' in locals():
            cap.release()

    if not coord_rows:
        return pd.DataFrame(columns=FRAME_META_COLUMNS + LANDMARK_COORD_COLUMNS)

    df_meta = pd.DataFrame(meta_rows, columns=FRAME_META_COLUMNS)
    df_coord = pd.DataFrame(coord_rows, columns=LANDMARK_COORD_COLUMNS)
    return pd.concat([df_meta, df_coord], axis=1)

@_timed
def get_undected_markers(frame: int, fps: int) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function creates a dataframe with NaN values representing facial landmarks that were not detected
    in a frame of the video.

    Parameters:
    ............
    frame : int
        Frame number
    fps : int
        Frames per second of the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe with NaN values for undetected facial landmarks in a frame of the video

    ---------------------------------------------------------------------------------------------------
    """
    return _get_empty_landmark_dataframe(frame, fps)

@_timed
def get_landmarks(path: str, bbox_list: BBoxList = []) -> pd.DataFrame:

    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to an image file and an error location string as input, and returns a Pandas
    dataframe containing the landmark coordinates for each frame of the video.

    Parameters:
    ............
    path : str
        Path to image file
    error_info : str
        Error location string
    bbox_list : list
        List of bounding boxes for each frame in the video

    Returns:
    ............
    df_landmark : pandas.DataFrame
        Dataframe containing the landmark coordinates for each frame of the video

    ---------------------------------------------------------------------------------------------------
    """

    cache_key = _get_landmark_cache_key(path, bbox_list)
    cached_df = _LANDMARK_CACHE.get(cache_key)

    if cached_df is not None:
        return cached_df

    df_landmark = run_facemesh(path, bbox_list=bbox_list)

    if len(df_landmark)>0:
        df_landmark = df_landmark.reset_index(drop=True)
        _LANDMARK_CACHE[cache_key] = df_landmark
    else:
        standard_fps = 25
        df_landmark = get_undected_markers(0,standard_fps)
        logger.info(f'Face not detected by mediapipe in file {path}')

    return df_landmark

@_timed
def get_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a Pandas dataframe of landmark coordinates as input, calculates the Euclidean distance
    between each landmark in consecutive frames, and returns a dataframe of the displacement values.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates

    Returns:
    ............
    displacement_df : pandas.DataFrame
        Dataframe containing displacement values for each landmark

    ---------------------------------------------------------------------------------------------------
    """
    disp_list = []

    for col in range(NUM_LANDMARKS):
        landmark_name = LANDMARK_DISPLACEMENT_COLUMNS[col]
        dist= np.sqrt(np.power(df[f'{landmark_name}_x'].shift() - df[f'{landmark_name}_x'], 2) +
                     np.power(df[f'{landmark_name}_y'].shift() - df[f'{landmark_name}_y'], 2) +
                     np.power(df[f'{landmark_name}_z'].shift() - df[f'{landmark_name}_z'], 2))

        df_dist = pd.DataFrame(dist, columns=[landmark_name])
        disp_list.append(df_dist)

    displacement_df = pd.concat(disp_list, axis=1).reset_index(drop=True)
    return displacement_df

@_timed
def get_mouth_height(df: pd.DataFrame, measures: Dict[str, Any]) -> pd.Series:
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a Pandas dataframe of landmark coordinates as input, calculates the Euclidean distance
    between the upper and lower lips, and returns an array of the displacement values.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    mouth_height : numpy.array
        Array of displacement values for mouth height

    ---------------------------------------------------------------------------------------------------
    """

    upper_lip_indices = measures["upper_lip_simple_landmarks"]
    lower_lip_indices = measures["lower_lip_simple_landmarks"]

    upper_lip = ['lmk' + str(col+1).zfill(3) for col in upper_lip_indices]
    lower_lip = ['lmk' + str(col+1).zfill(3) for col in lower_lip_indices]

    mouth_height = 0
    for i in [8, 9, 10]:
        mouth_height += np.sqrt(
            (df[upper_lip[i] + '_x'] - df[lower_lip[18-i] + '_x'])**2
            + (df[upper_lip[i] + '_y'] - df[lower_lip[18-i] + '_y'])**2
        )
    
    return mouth_height

@_timed
def get_lip_height(df: pd.DataFrame, lip: str, measures: Dict[str, Any]) -> pd.Series:
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a Pandas dataframe of landmark coordinates as input, calculates the Euclidean distance
    between the upper and lower parts of a lip, and returns an array of the displacement values.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates
    lip : str
        lip to calculate height for; must be either 'upper' or 'lower'
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    lip_height : numpy.array
        Array of displacement values for mouth height

    Raises:
    ............
    ValueError
        If lip is not 'upper' or 'lower'

    ---------------------------------------------------------------------------------------------------
    """

    lip = lip.lower()
    if lip not in ['upper', 'lower']:
        raise ValueError('lip must be either upper or lower')

    lip_indices = measures[f"{lip}_lip_simple_landmarks"]

    lip_landmarks = ['lmk' + str(col+1).zfill(3) for col in lip_indices]

    lip_height = 0
    for i in [2, 3, 4]:
        lip_height += np.sqrt(
            (df[lip_landmarks[i] + '_x'] - df[lip_landmarks[12-i] + '_x'])**2
            + (df[lip_landmarks[i] + '_y'] - df[lip_landmarks[12-i] + '_y'])**2
        )
    
    return lip_height

@_timed
def get_mouth_openness(df: pd.DataFrame, measures: Dict[str, Any]) -> pd.Series:
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates whether the mouth openness as the ratio of the mouth height to the min of
     upper lip and lower lip height.

    Parameters:
    ............
    df : pandas.DataFrame
        Dataframe containing landmark coordinates
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    mouth_openness : numpy.array
        Array of mouth openness values

    ---------------------------------------------------------------------------------------------------
    """

    upper_lip_height = get_lip_height(df, 'upper', measures)
    lower_lip_height = get_lip_height(df, 'lower', measures)
    mouth_height = get_mouth_height(df, measures)

    mouth_openness = mouth_height / np.minimum(upper_lip_height, lower_lip_height)

    return mouth_openness

@_timed
def baseline(
    base_path: str,
    bbox_list: BBoxList = [],
    normalize: bool = True,
    align: bool = False,
) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function takes a path to a baseline input file and returns a normalized Pandas dataframe of
    landmark coordinates.

    Parameters:
    ............
    base_path : str
        Path to baseline input file
    bbox_list : list
        List of bounding boxes for each frame in the video
    normalize : bool, optional
        Whether to normalize the facial landmarks to a common reference point (default is True)
    align : bool, optional
        Whether to align the facial landmarks based on the position of the eyes (default is False)

    Returns:
    ............
    base_df : pandas.DataFrame
        Normalized dataframe of landmark coordinates

    ---------------------------------------------------------------------------------------------------
    """

    base_landmark = get_landmarks(base_path, bbox_list=bbox_list)


    if normalize:
        base_df = normalize_face_landmarks(base_landmark, align=align)
        
    disp_base_df = get_distance(base_landmark)

    disp_base_df['overall'] = pd.DataFrame(disp_base_df.mean(axis=1))
    base_mean = disp_base_df.mean()

    base_df = pd.DataFrame(base_mean).T
    base_df = base_df[~base_df.isin([np.nan, np.inf, -np.inf])]

    base_df += 1 #Normalization
    return base_df

@_timed
def get_empty_dataframe() -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function creates an empty dataframe containing columns for frame number, landmark position
    variables, and overall displacement measurement.

    Parameters:
    ............
    None

    Returns:
    ............
    empty_df : pandas.DataFrame
        Empty displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """
    columns = FRAME_META_COLUMNS + LANDMARK_DISPLACEMENT_COLUMNS + ['overall']
    empty_df = pd.DataFrame(columns=columns)
    return empty_df

@_timed
def get_displacement(
        lmk_df: pd.DataFrame,
        base_path: str,
        measures: Dict[str, Any],
        base_bbox_list: BBoxList = [],
        normalize: bool = True,
        align: bool = False
    ) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the framewise euclidean displacement of each facial landmark from the landmark
    data and a given baseline. It returns a dataframe containing the framewise displacement data.

    Parameters:
    ............
    lmk_df : pandas.DataFrame
        facial landmark dataframe
    base_path : str
        baseline input file path
    measures : dict
        dictionary of landmark indices
    base_bbox_list : list, optional
        list of bounding boxes for each frame in the baseline video
    normalize : bool, optional
        whether to normalize the facial landmarks to a common reference point (default is True)
    align : bool, optional
        whether to align the facial landmarks based on the position of the eyes (default is False)

    Returns:
    ............
    displacement_df : pandas.DataFrame
         euclidean displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """

    displacement_df = get_empty_dataframe()

    try:
        df_meta = lmk_df[['frame','time']]

        if len(lmk_df)>1:
            disp_actual_df = get_distance(lmk_df)
            disp_actual_df['overall'] = pd.DataFrame(disp_actual_df.mean(axis=1))

            if os.path.exists(base_path):
                # add normalize flag
                disp_base_df = baseline(
                    base_path,
                    bbox_list=base_bbox_list,
                    normalize=normalize,
                    align=align
                )
                check_na = disp_base_df.iloc[:,1:].isna().all().all()

                if len(disp_base_df)> 0 and not check_na:
                    disp_actual_df = disp_actual_df + 1

                    disp_actual_df = disp_actual_df/disp_base_df.values
                    disp_actual_df = disp_actual_df - 1

            disp_actual_df = calculate_areas_displacement(disp_actual_df, measures)
            displacement_df = pd.concat([df_meta, disp_actual_df], axis=1).reset_index(drop=True)
    except Exception as e:

        logger.info(f'Error in displacement calculation is {e}')
    return displacement_df

@_timed
def calculate_areas_displacement(
    displacement_df: pd.DataFrame,
    measures: Dict[str, Any],
) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------

    This function calculates the summary framewise displacement for upper face,
     lower face, lips and eyebros.

    Parameters:
    ............
    displacement_df : pandas.DataFrame
        euclidean displacement dataframe
    measures : dict
        dictionary of landmark indices

    Returns:
    ............
    displacement_df : pandas.DataFrame
        updated euclidean displacement dataframe

    ---------------------------------------------------------------------------------------------------
    """

    lower_face_indices = measures["lower_face_landmarks"]
    upper_face_indices = [i for i in range(0, 468) if i not in lower_face_indices]
    lip_indices = measures["lips_landmarks"]
    eyebrow_indices = measures["eyebrows_landmarks"]

    lower_face_cols = ['lmk' + str(col+1).zfill(3) for col in lower_face_indices]
    upper_face_cols = ['lmk' + str(col+1).zfill(3) for col in upper_face_indices]
    lip_cols = ['lmk' + str(col+1).zfill(3) for col in lip_indices]
    eyebrow_cols = ['lmk' + str(col+1).zfill(3) for col in eyebrow_indices]

    displacement_df['lower_face'] = displacement_df[lower_face_cols].mean(axis=1)
    displacement_df['upper_face'] = displacement_df[upper_face_cols].mean(axis=1)
    displacement_df['lips'] = displacement_df[lip_cols].mean(axis=1)
    displacement_df['eyebrows'] = displacement_df[eyebrow_cols].mean(axis=1)

    return displacement_df

@_timed
def apply_rotation_per_frame(
    norm_df: pd.DataFrame,
    rotation_matrices: RotationMatrixList,
) -> pd.DataFrame:
    """
    Applies the corresponding rotation matrix to each row (frame) of the DataFrame.

    Parameters:
    norm_df (DataFrame): The DataFrame containing the centered landmarks.
    rotation_matrices (list of np.array): List of 3x3 rotation matrices for each frame.

    Returns:
    DataFrame: The rotated landmarks.
    """
    axes = ['x', 'y', 'z']
    landmark_cols = {
        axis: [col for col in norm_df.columns if col.endswith(f'_{axis}')]
        for axis in axes
    }

    stacked_coords = np.stack([
        norm_df[landmark_cols['x']].values,
        norm_df[landmark_cols['y']].values,
        norm_df[landmark_cols['z']].values
    ], axis=-1)  # Shape: (num_frames, num_landmarks, 3)

    rotated_coords = np.array([
        np.dot(stacked_coords[i], rotation_matrices[i].T) for i in range(len(rotation_matrices))
    ])

    for i, axis in enumerate(axes):
        norm_df[landmark_cols[axis]] = rotated_coords[..., i]

    return norm_df


@_timed
def calculate_rotation_matrix_for_all_frames(
    left_eye_x: pd.Series,
    left_eye_y: pd.Series,
    right_eye_x: pd.Series,
    right_eye_y: pd.Series,
) -> RotationMatrixList:
    """
    ---------------------------------------------------------------------------------------------------
    Calculates the rotation matrix for each frame based on eye positions to align the eyes horizontally.
    .................................
    Parameters:
    left_eye_x: int
        The x coordinate of the left eye.
    left_eye_y: int
        The y coordinate of the left eye.
    right_eye_x: int
        The x coordinate of the right eye.
    right_eye_y: int
        The y coordinate of the right eye.
    .................................
    Returns:
    list of np.array: List of 3x3 rotation matrices, one for each frame.
    ---------------------------------------------------------------------------------------------------
    """
    rotation_matrices = []
    
    for lx, ly, rx, ry in zip(left_eye_x, left_eye_y, right_eye_x, right_eye_y):
        delta_eye_x = lx - rx
        delta_eye_y = ly - ry

        # Compute 2D angle between eyes (ignore z-axis for horizontal alignment)
        theta = np.arctan2(delta_eye_y, delta_eye_x)

        # Create the rotation matrix to rotate landmarks around the Z-axis
        cos_theta = np.cos(-theta)  
        sin_theta = np.sin(-theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        rotation_matrices.append(rotation_matrix)

    return rotation_matrices

@_timed
def center_landmarks(df: pd.DataFrame, nose_tip: str) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------
    Centers the landmarks by moving the nose tip to the origin.

    Parameters:
    df (DataFrame):
        The DataFrame containing the landmarks.
    nose_tip (str): 
        The column name of the nose tip landmark.
    .................................
    Returns:
    DataFrame: The centered landmarks.
    ---------------------------------------------------------------------------------------------------
    """

    axes = ['x', 'y', 'z']
    nose_tip_coords = {axis: f'{nose_tip}_{axis}' for axis in axes}

    norm_data = {
        axis: df.filter(like=f'_{axis}') - df[nose_tip_coords[axis]].values[:, None]
        for axis in axes
    }

    norm_df = pd.concat(norm_data.values(), axis=1)
    return norm_df

@_timed
def get_vertices_for_col(df: pd.DataFrame, col_name: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extracts the x, y, and z coordinates for a given column.

    Parameters:
    df : DataFrame
         The DataFrame containing the columns.
    col_name : str
      The name of the column.

    Returns:
    x_col (Series): The x column.
    y_col (Series): The y column.
    z_col (Series): The z column.
    """
    x_col = df[f'{col_name}_x']
    y_col = df[f'{col_name}_y']
    z_col = df[f'{col_name}_z']

    return x_col, y_col, z_col

# Main function with the refactored parts included
@_timed
def normalize_face_landmarks(
    df: pd.DataFrame,
    align: bool = True,
    nose_tip: str = 'lmk001',
    left_eye: str = 'lmk144',
    right_eye: str = 'lmk373'
) -> pd.DataFrame:
    """
    ---------------------------------------------------------------------------------------------------
    Normalize the face landmarks by centering them around the nose tip and aligning the eyes horizontally.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing the face landmarks.
    align : bool, optional
        Whether to align the landmarks based on the position of the eyes (default is True).
    nose_tip : str, optional
        The name of the nose tip landmark (default is 'lmk001'). Note landmarks are 1-indexed in openwillis and 0-indexed in mediapipe
    left_eye : str, optional
        The name of the left eye landmark (default is 'lmk144').
    right_eye : str, optional
        The name of the right eye landmark (default is 'lmk373').

    Returns:
    --------
    DataFrame: The normalized face landmarks.
    ---------------------------------------------------------------------------------------------------
    """
    left_eye_x, left_eye_y, left_eye_z = get_vertices_for_col(df, left_eye)
    right_eye_x, right_eye_y, right_eye_z = get_vertices_for_col(df, right_eye)

    # compute the eye distance (for scaling)
    eye_distance = np.sqrt(
        (right_eye_x - left_eye_x)**2 +
        (right_eye_y - left_eye_y)**2 +
        (right_eye_z - left_eye_z)**2
    )

    scaling_factor =  eye_distance

    norm_df = center_landmarks(df, nose_tip)

    if align:

        rotation_matrices = calculate_rotation_matrix_for_all_frames(left_eye_x, left_eye_y, right_eye_x, right_eye_y)

        norm_df = apply_rotation_per_frame(norm_df, rotation_matrices)
    
     # scale the landmarks
    landmark_cols = [f'lmk{str(i).zfill(3)}_{axis}' for i in range(1, 469) for axis in ['x', 'y', 'z']]
    norm_df[landmark_cols] = norm_df[landmark_cols].div(scaling_factor.values, axis=0)

    norm_df[['frame','time']] = df[['frame','time']]

    return norm_df

def facial_expressivity(
    filepath: str,
    baseline_filepath: str = '',
    bbox_list: BBoxList = [],
    base_bbox_list: BBoxList = [],
    frames_per_second: Optional[int] = None,
    normalize: bool = True,
    align: bool = False,
    rolling_std_seconds: int = 3,
    split_by_speaking: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ---------------------------------------------------------------------------------------------------

    Uses mediapipe's facemesh solution to quantify the framewise 3D positioning of 468 facial landmarks.
    Calculates the framewise displacement of those landmarks to quantify movement in facial musculature
    as a proxy measure of overall facial expressivity.

    Parameters:
        filepath : str
            path to video
        baseline_filepath : str, optional
            optional path to baseline video. see openwillis research guidelines on github wiki to
            read case for baseline video use, particularly in clinical research contexts.
            (default is 0, meaning no baseline correction will be conducted).
        bbox_list : list, optional
            list of bounding boxes for each frame in the video. each bounding box is a dictionary
            with keys 'x', 'y', 'width', and 'height', representing the bounding box coordinates.
            (default is [], meaning no bounding boxes will be used).
        base_bbox_list : list, optional
            list of bounding boxes for each frame in the baseline video. each bounding box is a dictionary
            with keys 'x', 'y', 'width', and 'height', representing the bounding box coordinates.
            (default is [], meaning no bounding boxes will be used).
        frames_per_second : int, optional
            compatibility argument for external callers. the current implementation uses native video
            timing from frame metadata and does not require this value.
        normalize : bool, optional
            whether to normalize the facial landmarks to a common reference point (default is True).
        align : bool, optional
            whether to align the facial landmarks based on the position of the eyes (default is False).
        rolling_std_seconds : int, optional
            number of seconds over which to calculate the rolling standard deviation for speaking probability
        split_by_speaking : bool, optional
            whether to split the output by speaking probability (default is False).

    Returns:
        framewise_loc : pandas.DataFrame
            dataframe with framewise output of facial landmark 3D positioning. rows are frames in the input
            video. first column is frame number, second column is time in seconds, and all subsequent columns
            are landmark position variables, with each landmark numbered and further split into its x, y, and
            z coordinates. all coordinate values are between 0 and 1, relative to position in frame, as
            outputted by mediapipe.

        framewise_disp : pandas.DataFrame
            dataframe with framewise euclidean displacement of each facial landmark. rows are frames in input
            video (first row values are always zero). first column is frame number, second column is time in
            seconds, and subsequent columns are framewise displacement values for each facial landmark. last
            column is the overall framewisedisplacement measurement as a mean of all previous displacement columns.

        summary : pandas.DataFrame
            dataframe with summary measurements. first column is name of statistic, subsequent columns are all
            facial landmarks, last column is overall column with composite measures for all landmarks. first
            row contains sum of framewise displacement values, second row contains mean framewise displacement
            over the video, and third row has standard deviation of framewise displacement. in case an optional
            baseline video was provided, all summary measures are relative to baseline values calculated from
            baseline video.

    ---------------------------------------------------------------------------------------------------
    """
    config = get_config(os.path.abspath(__file__), "facial.json")
    if TIMING_ENABLED:
        _reset_timing_stats()
        start_timestamp = _current_timestamp()
        start_time = perf_counter()
        logger.info(
            "timing_start function=facial_expressivity timestamp=%s filepath=%s baseline_filepath=%s",
            start_timestamp,
            filepath,
            baseline_filepath,
        )
    else:
        start_time = None

    try:
        df_landmark = get_landmarks(filepath, bbox_list=bbox_list)
        
        if normalize:
            df_landmark = normalize_face_landmarks(df_landmark, align=align)
        df_disp = get_displacement(
            df_landmark,
            baseline_filepath,
            config,
            base_bbox_list=base_bbox_list,
            normalize=normalize,
            align=align
        )

        # use mouth height to calculate mouth openness
        df_disp['mouth_openness'] = get_mouth_openness(df_landmark, config)

        if split_by_speaking:
            df_disp['speaking_probability'] = get_speaking_probabilities(df_disp, rolling_std_seconds)
            df_summ = split_speaking_df(df_disp, 'speaking_probability', 470)

        else:
            df_summ = get_summary(df_disp, 470)

        return df_landmark, df_disp, df_summ

    except Exception as e:
        logger.info(f'Error in facial landmark calculation file: {filepath} & Error: {e}')
    finally:
        if TIMING_ENABLED and start_time is not None:
            elapsed_seconds = perf_counter() - start_time
            end_timestamp = _current_timestamp()
            _TIMING_STATS["facial_expressivity"]["count"] += 1
            _TIMING_STATS["facial_expressivity"]["total_seconds"] += elapsed_seconds
            logger.info(
                "timing_end function=facial_expressivity timestamp=%s elapsed_seconds=%.3f",
                end_timestamp,
                elapsed_seconds,
            )
            _log_timing_summary()
