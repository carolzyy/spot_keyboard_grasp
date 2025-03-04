import argparse
import sys
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
import os
import  pickle

def _draw_text_on_image(image, text):
    font_scale = 4
    thickness = 4
    font = cv2.FONT_HERSHEY_PLAIN
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale,
                                                thickness=thickness)[0]

    rectangle_bgr = (255, 255, 255)
    text_offset_x = 10
    text_offset_y = image.shape[0] - 25
    border = 10
    box_coords = ((text_offset_x - border, text_offset_y + border),
                  (text_offset_x + text_width + border, text_offset_y - text_height - border))
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale=font_scale,
                color=(0, 0, 0), thickness=thickness)


def add_grasp_constraint(config, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #   config.force_top_down_grasp:1
    #   config.force_horizontal_grasp : 2
    #   config.force_45_angle_grasp:3
    #   config.force_squeeze_grasp:4
    # You can specify more than one if you want and they will be OR'ed together.
    force_squeeze_grasp = False
    force_45_angle_grasp = False
    force_top_down_grasp = False
    force_horizontal_grasp = False
    if config==1:
        force_top_down_grasp = True
    elif config==2:
        force_horizontal_grasp = True
    elif config==3:
        force_45_angle_grasp = True
    elif config==4:
        force_squeeze_grasp = True


    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = force_top_down_grasp or force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


