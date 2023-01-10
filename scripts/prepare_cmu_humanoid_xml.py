import pathlib
import xml.etree.cElementTree as ET
from xml.dom import minidom

import requests
import numpy as np
# import urllib.request

DM_CONTROL_VERSION = "1.0.9"
XML_PATH = "dm_control/locomotion/walkers/assets/humanoid_CMU_V2020.xml"
XML_URL_TEMPLATE = (
  "https://raw.githubusercontent.com/deepmind/dm_control/"
  f"{{dm_control_version}}/{XML_PATH}")


def main():
  xml_url = XML_URL_TEMPLATE.format(dm_control_version=DM_CONTROL_VERSION)
  response = requests.get(xml_url)
  xml_string = response.text

  root_element = ET.fromstring(xml_string)
  root_element.set("model", "HumanoidCMU")

  default_element = root_element.find("default")
  assert default_element is not None
  humanoid_default_element = default_element.find("default[@class='humanoid']")
  assert humanoid_default_element is not None
  tracking_site_default = ET.SubElement(
    humanoid_default_element,
    "default",
    attrib={"class": "tracking_site"})
  _ = ET.SubElement(
    tracking_site_default,
    "site",
    type="sphere",
    size="0.027",
    rgba="1 0 0 1",
    group="3")

  sensor_element = root_element.find("sensor")
  assert sensor_element is not None
  root_element.remove(sensor_element)

  asset_element = root_element.find("asset")
  assert asset_element is not None
  root_element.remove(asset_element)

  world_body_element = root_element.find("worldbody")
  assert world_body_element is not None
  root_body_element = world_body_element.find("body[@name='root']")
  assert root_body_element is not None

  ET.SubElement(
    world_body_element,
    "geom",
    name="floor",
    type="plane",
    conaffinity="1",
    size="50 50 0.5",
    material="blue_grid")

  tracking_light = world_body_element.find("light[@name='tracking_light']")
  assert tracking_light is not None
  world_body_element.remove(tracking_light)
  root_body_element.insert(0, tracking_light)

  back_camera = world_body_element.find("camera[@name='back']")
  assert back_camera is not None
  world_body_element.remove(back_camera)
  root_body_element.insert(1, back_camera)

  side_camera = world_body_element.find("camera[@name='side']")
  assert side_camera is not None
  world_body_element.remove(side_camera)
  root_body_element.insert(2, side_camera)

  front_side_camera = world_body_element.find("camera[@name='front_side']")
  assert front_side_camera is not None
  world_body_element.remove(front_side_camera)
  root_body_element.insert(3, front_side_camera)

  root_geom = root_body_element.find("geom[@name='root_geom']")
  assert root_geom is not None
  root_geom_index = list(root_body_element).index(root_geom)
  free_joint = ET.Element("freejoint")
  root_body_element.insert(root_geom_index, free_joint)

  root_geom_thickness = root_geom.get("size").split(" ")[0]
  root_body_element.set("pos", f"0 0 {root_geom_thickness}")

  mocap_site_names_and_offsets = (
    # ("root", (0, 0, 0)),  # Root site already exists. See below.
    # ("head", (0, 0, 0.1)),  # Handle head manually. See below.
    ("ltoes", "ltoe", (0.0, -0.02, 0.0)),
    ("rtoes", "rtoe", (0.0, -0.02, 0.0)),
    ("lfoot", "lheel", (0.0, -0.030693, 0.0)),
    ("rfoot", "rheel", (0.0, -0.030693, 0.0)),
    ("ltibia", "lknee", (0.0, -0.02, 0.0)),
    ("rtibia", "rknee", (0.0, -0.02, 0.0)),
    ("lhand", "lhand", (0.0, 0.0, 0.0)),
    ("rhand", "rhand", (0.0, 0.0, 0.0)),
    ("lradius", "lelbow", (0.0, 0.013157, 0.0)),
    ("rradius", "relbow", (0.0, 0.013157, 0.0)),
    ("lhumerus", "lshoulder", (0.002, 0.0325, -0.004)),
    ("rhumerus", "rshoulder", (-0.002, 0.0325, -0.004)),
    ("lfemur", "lhip", (0.031937, 0.01, 0.039446)),
    ("rfemur", "rhip", (-0.031937, 0.01, 0.039446)),
  )

  root_site = root_body_element.find("site[@name='root']")
  assert root_site is not None
  root_site.set("size", "0.05")
  root_site.set("class", "tracking_site")
  root_site.set("pos", "0 -0.0121 0")
  del root_site.attrib["rgba"]

  head_body_element = root_body_element.find(".//body[@name='head']")
  head_site_element = ET.Element(
      "site",
      name="tracking[head]",
      pos=" ".join(list(map(str, (0, 0, 0.1)))),
      attrib={"class": "tracking_site"})
  head_body_element.insert(0, head_site_element)

  for mocap_site_name, tracking_site_name, mocap_site_offset in mocap_site_names_and_offsets:
    body_element = root_body_element.find(f".//body[@name='{mocap_site_name}']")
    assert body_element is not None
    body_element_parent = root_body_element.find(f".//body[@name='{mocap_site_name}']...")
    assert body_element_parent is not None

    body_pos = np.array(list(map(float, body_element.get("pos").split(" "))))
    mocap_site_pos = body_pos - mocap_site_offset

    mocap_site_element = ET.Element(
      "site",
      name=f"tracking[{tracking_site_name}]",
      pos=" ".join(list(map(str, mocap_site_pos.round(3)))),
      attrib={"class": "tracking_site"})
    mocap_site_index = list(body_element_parent).index(body_element)
    body_element_parent.insert(mocap_site_index, mocap_site_element)

  # For pybullet:
  # joint_elements = root_body_element.findall(f".//joint")
  # for joint_element in joint_elements:
  #   joint_element.set("type", "hinge")

  tree = ET.ElementTree(root_element)
  ET.indent(tree)

  output_str = ET.tostring(tree.getroot(), encoding="unicode", method="xml", short_empty_elements=True)
  output_str = output_str.replace(" /", "/")
  # output_str = minidom.parseString(ET.tostring(tree.getroot())).toprettyxml(indent="  ")

  output_path = pathlib.Path("/tmp/cmu_humanoid_mjpc.xml")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("wt") as f:
    f.write(output_str + "\n")


if __name__ == "__main__":
  main()
