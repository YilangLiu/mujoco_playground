from etils import epath

from mujoco_playground._src import mjx_env

def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_SITES = [
    "FR",
    "FL",
    "RR",
    "RL",
]

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go2"

MJX_XML_PATH = (
    ROOT_PATH / "xmls" / "scene_mjx_collision_free.xml"
)

MUJOCO_XML_PATH = (
    ROOT_PATH / "xmls" / "scene.xml"
)

ONNX_DIR = mjx_env.ROOT_PATH / "experimental" / "sim2sim" / "onnx"

FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
# Go2SampleAPG-specific rough scene: feet-only robot + hfield, but with the
# flat scene's home keyframe (action_loc / reference compatibility).
SAMPLE_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_sample_rough_terrain.xml"
)
# Go2SampleAPG crate-climbing scene: dial-mpc's 0.6 m crate + a torso
# collision box (the sampled climb rests the belly on the crate edge).
SAMPLE_CRATE_XML = ROOT_PATH / "xmls" / "scene_mjx_sample_crate.xml"
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"