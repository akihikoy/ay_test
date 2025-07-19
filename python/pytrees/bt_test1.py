#!/usr/bin/python3
#\file    bt_test1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.19, 2025

import random
import py_trees

# Blackboard keys
BATTERY_LEVEL_KEY = "battery_level"
PERSON_NEAR_KEY = "person_near"
OTHERROBOT_NEAR_KEY = "otherrobot_near"
ROOM_LIST_KEY = "rooms"
CURRENT_ROOM_KEY = "current_room"

class UpdateSensors(py_trees.behaviour.Behaviour):
  """Update sensor data on the blackboard (battery and person detection)."""
  def __init__(self, name="UpdateSensors"):
    super().__init__(name)
  def update(self):
    bb = py_trees.blackboard.Blackboard()
    # fetch battery level, defaulting to 100 if not set
    level = bb.get(BATTERY_LEVEL_KEY)
    level = 100 if level is None else level
    new_level = max(0, level - random.randint(10, 25))
    bb.set(BATTERY_LEVEL_KEY, new_level)
    # randomly decide if a person is near
    bb.set(PERSON_NEAR_KEY, random.random() < 0.2)
    # randomly decide if another robot is near
    bb.set(OTHERROBOT_NEAR_KEY, random.random() < 0.2)
    return py_trees.common.Status.SUCCESS

class BatteryLow(py_trees.behaviour.Behaviour):
  """Check if the battery is below a threshold."""
  def __init__(self, threshold=30, name="BatteryLow"):
    super().__init__(name)
    self.threshold = threshold
  def update(self):
    bb = py_trees.blackboard.Blackboard()
    level = bb.get(BATTERY_LEVEL_KEY)
    # If level is not yet set, treat as full battery (not low)
    return py_trees.common.Status.SUCCESS if level is not None and level < self.threshold else py_trees.common.Status.FAILURE

class GoCharge(py_trees.behaviour.Behaviour):
  """Simulate going to the charging dock."""
  def __init__(self, name="GoCharge"):
    super().__init__(name)
  def update(self):
    bb = py_trees.blackboard.Blackboard()
    bb.set(BATTERY_LEVEL_KEY, 100)
    self.feedback_message = "Charging completed"
    return py_trees.common.Status.SUCCESS

class PersonNear(py_trees.behaviour.Behaviour):
  """Check if a person is near the robot."""
  def __init__(self, name="PersonNear"):
    super().__init__(name)
  def update(self):
    return py_trees.common.Status.SUCCESS if py_trees.blackboard.Blackboard().get(PERSON_NEAR_KEY) else py_trees.common.Status.FAILURE

class SafeStop(py_trees.behaviour.Behaviour):
  """Take a safety action when someone is near."""
  def __init__(self, name="SafeStop"):
    super().__init__(name)
  def update(self):
    self.feedback_message = "Person detected, stopping"
    return py_trees.common.Status.SUCCESS

class OtherRobotNear(py_trees.behaviour.Behaviour):
  """Check if another robot is near the robot."""
  def __init__(self, name="OtherRobotNear"):
    super().__init__(name)
  def update(self):
    return py_trees.common.Status.SUCCESS if py_trees.blackboard.Blackboard().get(OTHERROBOT_NEAR_KEY) else py_trees.common.Status.FAILURE

class YieldToRobot(py_trees.behaviour.Behaviour):
  """Take an action to yield to the robot when another robot is near."""
  def __init__(self, name="YieldToRobot"):
    super().__init__(name)
  def update(self):
    self.feedback_message = "Another robot detected, yielding"
    return py_trees.common.Status.SUCCESS

class PatrolRooms(py_trees.behaviour.Behaviour):
  """Move to the next room in the list."""
  def __init__(self, name="PatrolRooms"):
    super().__init__(name)
  def update(self):
    bb = py_trees.blackboard.Blackboard()
    rooms = bb.get(ROOM_LIST_KEY) or []
    index = bb.get(CURRENT_ROOM_KEY) or 0
    if index >= len(rooms):
      self.feedback_message = "All rooms visited"
      return py_trees.common.Status.SUCCESS
    room = rooms[index]
    self.feedback_message = f"Moving to {room}"
    # advance to the next room
    bb.set(CURRENT_ROOM_KEY, index + 1)
    return py_trees.common.Status.RUNNING

def create_tree():
  """Construct the behaviour tree for the mobile robot."""
  # Selector with memory disabled so it reevaluates children every tick
  root = py_trees.composites.Selector(name="Root", memory=False)
  # Battery management subtree
  battery_sequence = py_trees.composites.Sequence(name="BatteryCheck", memory=False)
  battery_sequence.add_children([BatteryLow(threshold=30), GoCharge(name="Recharge")])
  # Safety subtree
  safety_sequence = py_trees.composites.Sequence(name="SafetyCheck", memory=False)
  safety_sequence.add_children([PersonNear(name="DetectPerson"), SafeStop(name="Stop")])
  # Yield subtree
  yield_sequence = py_trees.composites.Sequence(name="OtherRobotCheck", memory=False)
  yield_sequence.add_children([OtherRobotNear(name="DetectOtherRobot"), YieldToRobot(name="Yield")])
  # Patrol subtree
  patrol = PatrolRooms(name="PatrolRooms")
  # Add subtrees in priority order
  root.add_children([battery_sequence, safety_sequence, yield_sequence, patrol])
  # Create the behaviour tree and register a pre-tick sensor updater
  tree = py_trees.trees.BehaviourTree(root)
  updater = UpdateSensors()
  tree.add_pre_tick_handler(lambda tree: updater.update())
  return tree

# Initialise blackboard values
bb = py_trees.blackboard.Blackboard()
bb.set(BATTERY_LEVEL_KEY, 100)
bb.set(PERSON_NEAR_KEY, False)
bb.set(ROOM_LIST_KEY, ["RoomA", "RoomB", "RoomC"])
bb.set(CURRENT_ROOM_KEY, 0)

# Create and run the tree
bt = create_tree()

print('Behavior tree:')
# Show as ascii/unicode text
#print(py_trees.display.ascii_tree(bt.root))
print(py_trees.display.unicode_tree(bt.root))

# Save as SVG via DOT
#py_trees.display.dot_tree(bt.root).write_svg("tree.svg")

# Render tool
py_trees.display.render_dot_tree(
  bt.root,
  target_directory=".",
  name="tree",
  with_blackboard_variables=False,
  collapse_decorators=False
)

# Save as XHTML.
html = py_trees.display.xhtml_tree(bt.root)
with open("tree.html", "w") as fp:
  fp.write(html)

for i in range(10):
  print(f"\nTick {i+1}")
  bt.tick()
  # print statuses with ascii tree for debugging
  #print(py_trees.display.ascii_tree(bt.root, show_status=True))
  print(py_trees.display.unicode_tree(bt.root, show_status=True))
  print(py_trees.display.unicode_blackboard())

