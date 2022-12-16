import pathlib
import xml.etree.cElementTree as ET
from xml.dom import minidom

import requests
# import urllib.request

DM_CONTROL_VERSION = "1.0.9"
XML_PATH = "dm_control/locomotion/walkers/assets/humanoid_CMU_V2020.xml"
XML_URL_TEMPLATE = f"https://raw.githubusercontent.com/deepmind/dm_control/{{dm_control_version}}/{XML_PATH}"

# "https://raw.githubusercontent.com/deepmind/dm_control/1.0.9/dm_control/locomotion/walkers/assets/humanoid_CMU_V2020.xml"
# "https://raw.githubusercontent.com/deepmind/dm_control/4e1a35595124742015ae0c7a829e099a5aa100f5/dm_control/locomotion/walkers/assets/humanoid_CMU_V2020.xml"

def main():
  xml_url = XML_URL_TEMPLATE.format(dm_control_version=DM_CONTROL_VERSION)
  response = requests.get(xml_url)
  xml_string = response.text

  root_element = ET.fromstring(xml_string)
  root_element.set("model", "HumanoidCMU")

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

  mocap_site_names_and_offsets = (
    # ("root", (0, 0, 0)),
    ("head", (0, 0, 0.1)),
    ("ltoes0", (0, 0, 0)),
    ("rtoes0", (0, 0, 0)),
    ("lfoot", (0, 0, 0.05)),
    ("rfoot", (0, 0, 0.05)),
    ("ltibia", (0, 0, 0)),
    ("rtibia", (0, 0, 0)),
    ("lwrist", (0, 0, 0)),
    ("rwrist", (0, 0, 0)),
    ("lradius", (0, 0, 0)),
    ("rradius", (0, 0, 0)),
    ("lhumerus", (0, 0, 0)),
    ("rhumerus", (0, 0, 0)),
    ("lfemur", (0, 0, 0)),
    ("rfemur", (0, 0, 0)),
  )

  root_site = root_body_element.find("site[@name='root']")
  assert root_site is not None
  root_site.set("size", "0.027")
  root_site.set("rgba", "1 0 0 1")
  root_site.set("group", "3")

  for mocap_site_name, mocap_site_offset in mocap_site_names_and_offsets:
    mocap_site_element = ET.Element(
      "site",
      name=f"tracking-{mocap_site_name}",
      type="sphere",
      pos=" ".join(map(str, mocap_site_offset)),
      size="0.027",
      rgba="1 0 0 1",
      group="3")
    # mocap_body_element = ET.Element(
    #   "body",
    #   name=f"tracking-{mocap_site_name}",
    #   pos=" ".join(map(str, mocap_site_offset)))
    geom_parent_element = root_body_element.find(f".//geom[@name='{mocap_site_name}']...")
    assert geom_parent_element is not None
    # geom_element = root_body_element.find(f".//geom[@name='{mocap_site_name}']")
    # geom_index = list(geom_parent_element).index(geom_element)
    # mocap_site_index = geom_index + 1
    mocap_site_index = 0
    geom_parent_element.insert(mocap_site_index, mocap_site_element)
    # geom_parent_element.insert(mocap_site_index, mocap_body_element)

  # For pybullet:
  # joint_elements = root_body_element.findall(f".//joint")
  # for joint_element in joint_elements:
  #   joint_element.set("type", "hinge")

  tree = ET.ElementTree(root_element)
  ET.indent(tree)

  output_str = ET.tostring(tree.getroot(), encoding="unicode", method="xml", short_empty_elements=True)
  output_str = output_str.replace(" /", "/")
  # output_str = minidom.parseString(ET.tostring(tree.getroot())).toprettyxml(indent="  ")

  output_path = pathlib.Path("/tmp/humanoid_CMU_mjpc.xml")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("wt") as f:
    f.write(output_str)


if __name__ == "__main__":
  main()
