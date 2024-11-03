import os
import numpy as np
import sys
import sqlite3
from xvfbwrapper import Xvfb


IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def pipeline(scene, base_path, n_views):
    view_path = str(n_views) + '_views'
    scene_path = os.path.join(base_path, scene)
    os.chdir(scene_path)

    # Remove and recreate view_path directory
    os.system('rm -rf ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')

    # Copy images from 'train' directory to 'images' directory
    os.system('cp ../train/*.* images/')

    with Xvfb() as xvfb:
        # Feature extraction
        os.system(
            'colmap feature_extractor '
            '--database_path database.db '
            '--image_path images '
            '--SiftExtraction.max_image_size 4032 '
            '--SiftExtraction.max_num_features 32768 '
            '--SiftExtraction.use_gpu 0'
        )
        # Feature matching
        os.system(
            'colmap exhaustive_matcher '
            '--database_path database.db '
            '--SiftMatching.guided_matching 0 '
            '--SiftMatching.max_num_matches 32768 '
            '--SiftMatching.use_gpu 0'
        )
        # Mapping (structure from motion)
        os.system('colmap mapper --database_path database.db --image_path images --output_path sparse')

        # Convert model to TXT format
        os.system('colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT')

    # Read images and their metadata from images.txt
    images = {}
    with open('sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                fid.readline()
                images[image_name] = elems[1:]

    img_list = sorted(images.keys())
    # Select n_views images
    if n_views > 0:
        idx_sub = [int(round(i)) for i in np.linspace(0, len(img_list)-1, n_views)]
        train_img_list = [img_list[idx] for idx in idx_sub]
    else:
        train_img_list = img_list

    # Clean images directory and copy selected images
    os.system('rm -rf images')
    os.mkdir('images')
    for img_name in train_img_list:
        os.system('cp ../train/' + img_name + ' images/' + img_name)

    with Xvfb() as xvfb:
        # Re-run feature extraction on selected images
        os.system(
            'colmap feature_extractor '
            '--database_path database_selected.db '
            '--image_path images '
            '--SiftExtraction.max_image_size 4032 '
            '--SiftExtraction.max_num_features 32768 '
            '--SiftExtraction.use_gpu 0'
        )
        # Re-run matching on selected images
        os.system(
            'colmap exhaustive_matcher '
            '--database_path database_selected.db '
            '--SiftMatching.guided_matching 0 '
            '--SiftMatching.max_num_matches 32768 '
            '--SiftMatching.use_gpu 0'
        )
        # Reconstruct model with selected images
        os.system('colmap mapper --database_path database_selected.db --image_path images --output_path triangulated')

        # Convert model to TXT format
        os.system('colmap model_converter --input_path triangulated/0 --output_path triangulated/0 --output_type TXT')

    # Undistort images and prepare for dense reconstruction
    os.system('colmap image_undistorter --image_path images --input_path triangulated/0 --output_path dense')
    os.system('colmap patch_match_stereo --workspace_path dense')
    os.system('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')

# Adjust the base path to your dataset's location
base_path = '/pscratch/sd/j/jinchuli/drProject/nerf_synthetic/'

for scene in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
    pipeline(scene, base_path=base_path, n_views=3)