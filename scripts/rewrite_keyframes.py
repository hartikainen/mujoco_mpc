import pathlib
import xml.etree.cElementTree as ET
from xml.dom import minidom
import requests

import numpy as np


def rewrite_frames_for_xml(filename):
  xml_path = pathlib.Path(
      "~/github/deepmind/mujoco_mpc/mjpc/tasks/humanoid_cmu/track_sequence/keyframes",
    filename
  ).expanduser()
  xml_string = xml_path.read_text()


  mujoco_element = ET.fromstring(xml_string)
  keyframe_element = mujoco_element.find("keyframe")
  assert keyframe_element is not None

  # key_elements = keyframe_element.findall("key")
  # mposes = []
  # for key_element in key_elements:
  #     mpos_str = key_element.get("mpos")
  #     mpos = list((map(float, mpos_str.split(" "))))
  #     mposes.append(mpos)

  # mposes = np.array(mposes)
  # min_mposes_z = mposes[..., 2::3].min()

  # if np.abs(min_mposes_z) < 1e-3:
  #     return
  # mposes[..., 2::3] -= min_mposes_z

  home_key_element = keyframe_element.find("key[@name='home']")
  mpos_str = home_key_element.get("mpos")
  mpos = list((map(float, mpos_str.split(" "))))
  qpos_str = home_key_element.get("qpos")
  qpos = list((map(float, qpos_str.split(" "))))
  qpos[2] = mpos[2]
  home_key_element.set("qpos", " ".join(map(str, qpos)))

  tree = ET.ElementTree(mujoco_element)
  ET.indent(tree)

  output_str = ET.tostring(tree.getroot(), encoding="unicode", method="xml", short_empty_elements=True)
  # output_str = output_str.replace(" /", "/")
  # output_str = minidom.parseString(ET.tostring(tree.getroot())).toprettyxml(indent="  ")

  output_path = xml_path
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("wt") as f:
    f.write(output_str)


def main():
  filenames = (
      "CMU-CMU-02-02_04_poses.xml",
      "CMU-CMU-103-103_08_poses.xml",
      "CMU-CMU-108-108_13_poses.xml",
      "CMU-CMU-137-137_40_poses.xml",
      "CMU-CMU-87-87_01_poses.xml",
      "CMU-CMU-88-88_06_poses.xml",
      "CMU-CMU-88-88_07_poses.xml",
      "CMU-CMU-88-88_08_poses.xml",
      "CMU-CMU-88-88_09_poses.xml",
      "CMU-CMU-90-90_19_poses.xml",
      "HumanEva-HumanEva-S3-Static_poses.xml",
      "KIT-KIT-1229-open_pants_right_arm_03_poses.xml",
  )
  for filename in filenames:
      rewrite_frames_for_xml(filename)


if __name__ == "__main__":
  main()
