'''
Date: 2024.02.09
Update: change image save
    the number and name of all the file in image folder are same
'''

import bosdyn.client.util
from bosdyn.client.async_tasks import  AsyncPeriodicGRPCTask
from bosdyn.geometry import EulerZXY
import pickle
import curses
import io
import logging
import signal
import threading
from PIL import Image
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.estop import  EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME,HAND_FRAME_NAME,
                                         get_a_tform_b, get_vision_tform_body)
from bosdyn.client import math_helpers

from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, secs_to_hms

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,)
from bosdyn.client.robot_state import RobotStateClient

import traceback
from datetime import datetime
import os
import numpy as np   
import time
from scipy import ndimage
from bosdyn.client.image import ImageClient
from bosdyn.api.image_pb2 import ImageSource
from bosdyn.client.manipulation_api_client import ManipulationApiClient
import torch
import utils as u
import ImageProcess as imgutil
import cv2
LOGGER = logging.getLogger(__name__)

path = '/home/carol/Project/Spot/grasp_in_image/Spot-Reach-v0/102/reach'
#/home/carol/Project/Spot/grasp_in_image/Spot-Reach-v0/102/reach_afford_new/final_ddqn.pth
afford=True
prior=True
model='_new/final_ddqn.pth'
if prior:
    model ='_prior'+model
if afford:
    model = '_afford' + model
load_path = path+model
from DQN_play import DDQN
agent = DDQN(
    afford=afford,
    prior=prior,
)
agent.load_model(path=load_path)


# Logs name
# datetime object containing current date and time
# dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
base_path = 'logs-aff/'+dt_string+'/'+model.split('/')[0]
STATE_LOG_PATH=base_path+'/state/'
os.makedirs(STATE_LOG_PATH, exist_ok=True)


UNLOCK=False # if False to lock the arm movement in world coordinates
AI_CONTROL=False # if False to human input as control

VELOCITY_BASE_SPEED = 0.45  # m/s
VELOCITY_BASE_ANGULAR = 0.75  # rad/sec
VELOCITY_CMD_DURATION = 0.4  # seconds
COMMAND_INPUT_RATE = 0.1

#Robot Arm Movement
VELOCITY_HAND_NORMALIZED = 0.5  # normalized hand velocity [0,1]
VELOCITY_ANGULAR_HAND = 1.0  # rad/sec

#Buffer to store some previous states/imgs
PREV_IMG = None
PREV_STATE = None

# Mapping from visual to depth data
VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE = {
    'frontleft_fisheye_image': 'frontleft_depth_in_visual_frame',
    'frontright_fisheye_image': 'frontright_depth_in_visual_frame'
}
ROTATION_ANGLES = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


class ExitCheck(object):
    """A class to help exiting a loop, also capturing SIGTERM to exit the loop."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    def request_exit(self):
        """Manually trigger an exit (rather than sigterm/sigint)."""
        self._kill_now = True

    @property
    def kill_now(self):
        """Return the status of the exit checker indicating if it should exit."""
        return self._kill_now

class CursesHandler(logging.Handler):
    """logging handler which puts messages into the curses interface"""

    def __init__(self, arm_wasd_interface):
        super(CursesHandler, self).__init__()
        self._arm_wasd_interface = arm_wasd_interface

    def emit(self, record):
        msg = record.getMessage()
        msg = msg.replace('\n', ' ').replace('\r', '')
        self._arm_wasd_interface.add_message(f'{record.levelname:s} {msg:s}')

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.0001)

    def _start_query(self):
        return self._client.get_robot_state_async()
class AsyncArmStateCapture(AsyncPeriodicQuery):

    def __init__(self, robot_state_client):
        super(AsyncArmStateCapture, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.0001)

    def _start_query(self):
        return self._client.get_robot_state_async()

    def _should_query(self, now_sec):  # pylint: disable=unused-argument
        return (now_sec - self._last_call) > self._period_sec

    def _handle_result(self, result):
        try:
            #timestamp = result.kinematic_state.acquisition_timestamp.seconds + \
            #            result.kinematic_state.acquisition_timestamp.nanos * 1e-9
            timestamp = "%.20f" % time.time()

            arm_filename = STATE_LOG_PATH + str(timestamp)+'.pkl'

            with open(arm_filename, 'wb') as f:
                pickle.dump(result, f)

        except Exception as e:
            LOGGER.exception('Error saving the image: %s', e)

    def _handle_error(self, exception):
        LOGGER.exception('Failure getting image: %s', exception)

class Image_Process():
    """Grab camera images from the robot."""

    def __init__(self, encode_flag=True,sgm_flag=True):
        if encode_flag:
            self.encoder = imgutil.DinoProcessor()
        if sgm_flag:
            self.segmentor = imgutil.SamProcessor()
        self.encode_flag = encode_flag
        self.sgm_flag = sgm_flag



class WasdInterface(object):
    """A curses interface for driving the robot."""

    def __init__(self, robot):
        self._robot = robot
        # Create clients -- do not use the for communication yet.
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        except:
            # Not the estop.
            self._estop_client = None
            self._estop_endpoint = None
        self._power_client = robot.ensure_client(PowerClient.default_service_name)

        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_mani_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        self._arm_log_task = AsyncArmStateCapture(self._robot_state_client)
        #['frontright_fisheye_image', 'frontleft_fisheye_image', 'frontright_depth', 'frontleft_depth']
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._async_tasks = AsyncTasks([self._robot_state_task,
                                        self._arm_log_task,
                                        #self._image_task
                                        ])

        self.image_source = ['frontright_fisheye_image']
        depth_source = VISUAL_SOURCE_TO_DEPTH_MAP_SOURCE[self.image_source[0]]
        self.image_source.append(depth_source)
        self.image=None
        self.grasp_position = None
        self.imgprocessor=Image_Process(encode_flag=True,sgm_flag=True)


        self._lock = threading.Lock()
        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            ord(' '): self._toggle_estop,
            ord('p'): self._toggle_power,
            ord('c'): self._sit,
            ord('f'): self._stand,
            ord('w'): self._move_forward,
            ord('s'): self._move_backward,
            ord('a'): self._strafe_left,
            ord('d'): self._strafe_right,
            ord('N'): self._toggle_gripper_open,
            ord('M'): self._toggle_gripper_closed,
            ord('q'): self._turn_left,
            ord('e'): self._turn_right,
            #ord('i'): self._toggle_image_capture,
            ord('y'): self._unstow,
            ord('h'): self._stow,
            ord('v'): self._show_image_grasp,
            ord('k'): self._grasp_from_agent,
            ord('.'): self.reset_AI_CONTROL,
            
        }
        self._locked_messages = ['', '', '']  # string: displayed message for user
        self._estop_keepalive = None
        self._exit_check = None

        # Stuff that is set in start()
        self._robot_id = None
        self._lease_keepalive = None


    def start(self):
        """Begin communication with the robot."""
        # Construct our lease keep-alive object, which begins RetainLease calls in a thread.
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                               return_at_exit=True)

        self._robot_id = self._robot.get_id()
        if self._estop_endpoint is not None:
            self._estop_endpoint.force_simple_setup(
            )  # Set this endpoint as the robot's sole estop.

    def shutdown(self):
        """Release control of robot as gracefully as possible."""
        LOGGER.info('Shutting down WasdInterface.')
        if self._estop_keepalive:
            # This stops the check-in thread but does not stop the robot.
            self._estop_keepalive.shutdown()
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()

    def flush_and_estop_buffer(self, stdscr):
        """Manually flush the curses input buffer but trigger any estop requests (space)"""
        key = ''
        while key != -1:
            key = stdscr.getch()
            if key == ord(' '):
                self._toggle_estop()
                self._toggle_power()

    def add_message(self, msg_text):
        """Display the given message string to the user in the curses interface."""
        with self._lock:
            self._locked_messages = [msg_text] + self._locked_messages[:-1]

    def message(self, idx):
        """Grab one of the 3 last messages added."""
        with self._lock:
            return self._locked_messages[idx]

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto

    def drive(self, stdscr):
        """User interface to control the robot via the passed-in curses screen interface object."""
        global PREV_IMG
        global AI_CONTROL
        with ExitCheck() as self._exit_check:
            curses_handler = CursesHandler(self)
            curses_handler.setLevel(logging.INFO)
            LOGGER.addHandler(curses_handler)

            stdscr.nodelay(True)  # Don't block for user input.
            stdscr.resize(26, 140)
            stdscr.refresh()

            # for debug
            curses.echo()

            try:
                while not self._exit_check.kill_now:
                    self._async_tasks.update()
                    self._drive_draw(stdscr, self._lease_keepalive)

                    if not AI_CONTROL:
                        try:
                            cmd = stdscr.getch()
                            # Do not queue up commands on client
                            self.flush_and_estop_buffer(stdscr)
                            LOGGER.info(f'action is{stdscr}')
                            self._drive_cmd(cmd)
                            time.sleep(COMMAND_INPUT_RATE)
                        except Exception:
                            # On robot command fault, sit down safely before killing the program.
                            print(traceback.format_exc())
                            self._safe_power_off()
                            time.sleep(2.0)
                            raise
                    else:
                        try:
                            state = self._robot_state_client.get_robot_state()
                        except Exception:
                            # On robot command fault, sit down safely before killing the program.
                            print(traceback.format_exc())
                            self._safe_power_off()
                            time.sleep(2.0)
                            raise


            finally:
                LOGGER.removeHandler(curses_handler)

    def _drive_draw(self, stdscr, lease_keep_alive):
        """Draw the interface screen at each update."""
        stdscr.clear()  # clear screen
        stdscr.resize(28, 140)
        stdscr.addstr(0, 0, f'{self._robot_id.nickname:20s} {self._robot_id.serial_number}')
        stdscr.addstr(1, 0, self._lease_str(lease_keep_alive))
        stdscr.addstr(2, 0, self._battery_str())
        stdscr.addstr(3, 0, self._estop_str())
        stdscr.addstr(4, 0, self._power_state_str())
        stdscr.addstr(5, 0, self._time_sync_str())
        for i in range(3):
            stdscr.addstr(7 + i, 2, self.message(i))
        stdscr.addstr(10, 0, '          Commands: [TAB]: quit                                  ')
        stdscr.addstr(11, 0, '          [i]: Take image and show')
        stdscr.addstr(12, 0, '          [f]: Stand,                ')
        stdscr.addstr(13, 0, '          [c]: Sit,                ')
        stdscr.addstr(14, 0, '          [y]: Unstow arm, [h]: Stow arm               ')
        stdscr.addstr(15, 0, '          [wasd]: Directional strafing                 ')
        stdscr.addstr(16, 0, '          [NM]: Open/Close gripper                  ')
        stdscr.addstr(17, 0, '          [qe]: Body Turning, [ESC]: Stop                   ')
        stdscr.addstr(18, 0, '          [l]: Return/Acquire lease                  ')
        stdscr.addstr(19, 0, '          [v],show mask of the image')
        stdscr.addstr(20, 0, '          [m],detect the target and move to it automatically')


        global AI_CONTROL
        global UNLOCK

        if AI_CONTROL:
            stdscr.addstr(26, 0, 'Now AI control mode')
        else:
            stdscr.addstr(26, 0, 'Now Human control mode') 


        if UNLOCK:
            stdscr.addstr(27, 0, 'Now hand is unlocked in space')
        else:
            stdscr.addstr(27, 0, 'Now hand is locked in space')

        stdscr.refresh()

    def _drive_cmd(self, key):
        """Run user commands at each update."""
        try:
            cmd_function = self._command_dictionary[key]
            cmd_function()

        except KeyError:
            if key and key != -1 and key < 256:
                self.add_message(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _try_grpc(self, desc, thunk):
        try:
            cmd_id = thunk()
            state = self._robot_state_client.get_robot_state
            #self.robot_state_handler_logger.info(state)
            return cmd_id
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self.add_message(f'Failed {desc}: {err}')
            return None

    def _try_grpc_async(self, desc, thunk):

        def on_future_done(fut):
            try:
                fut.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                self.add_message(f'Failed {desc}: {err}')
                return None

        future = thunk()
        future.add_done_callback(on_future_done)

    def _quit_program(self):
        self._sit()
        if self._exit_check is not None:
            self._exit_check.request_exit()


    def _toggle_estop(self):
        """toggle estop on/off. Initial state is ON"""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    def _toggle_lease(self):
        """toggle lease acquisition. Initial state is acquired"""
        if self._lease_client is not None:
            if self._lease_keepalive is None:
                self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                                       return_at_exit=True)
            else:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None

    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            return self._robot_command_client.robot_command(command=command_proto,
                                                     end_time_secs=end_time_secs)

        return self._try_grpc(desc, _start_command)


    def _sit(self):
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())
        #self._toggle_image_capture()


    def _move_forward(self):
        self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

    def _move_backward(self):
        self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

    def _strafe_left(self):
        self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

    def _strafe_right(self):
        self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

    def _arm_move_out(self):
        self._arm_cylindrical_velocity_cmd_helper('move_out', v_r=VELOCITY_HAND_NORMALIZED)

    def _arm_move_in(self):
        self._arm_cylindrical_velocity_cmd_helper('move_in', v_r=-VELOCITY_HAND_NORMALIZED)

    def _rotate_ccw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_ccw', v_theta=VELOCITY_HAND_NORMALIZED)

    def _rotate_cw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_cw', v_theta=-VELOCITY_HAND_NORMALIZED)

    def _rotate_plus_rx(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rx(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_ry(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_ry(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_rz(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rz(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)
        else:
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)


    def _arm_cartesian_move_out(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_out_cartesian', v_x=0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_out', v_r=VELOCITY_HAND_NORMALIZED)


    def _arm_cartesian_move_in(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_in_cartesian', v_x=-0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_in', v_r=-VELOCITY_HAND_NORMALIZED)


    def _rotate_cartesian_ccw(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('rotate_ccw_cartesian', v_y=0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('rotate_ccw', v_theta=VELOCITY_HAND_NORMALIZED)

    def _rotate_cartesian_cw(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('rotate_cw_cartesian', v_y=-0.2)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('rotate_cw', v_theta=-VELOCITY_HAND_NORMALIZED)


    def _move_up(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_up', v_z=VELOCITY_HAND_NORMALIZED)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_up', v_z=VELOCITY_HAND_NORMALIZED)


    def _move_down(self):
        if UNLOCK:
            self._arm_cartesian_velocity_cmd_helper('move_down', v_z=-VELOCITY_HAND_NORMALIZED)
        else:
            self._arm_cylindrical_velocity_cmd_helper_LOCK('move_down', v_z=-VELOCITY_HAND_NORMALIZED)


    def _toggle_gripper_open(self):
        self._start_robot_command('open_gripper', RobotCommandBuilder.claw_gripper_open_command())

    def _toggle_gripper_closed(self):
        self._start_robot_command('close_gripper', RobotCommandBuilder.claw_gripper_close_command())


    def reset_AI_CONTROL(self):
        global AI_CONTROL
        AI_CONTROL = True
        print('now AI control')

    def reset_HUMAN_CONTROL(self):
        global AI_CONTROL
        AI_CONTROL = False

        print('now Human control')


    def _orientation_cmd_helper(self, yaw=0.0, roll=0.0, pitch=0.0, height=0.0):
        """Helper function that commands the robot with an orientation command;
        Used by the other orientation functions.

        Args:
            yaw: Yaw of the robot body. Defaults to 0.0.
            roll: Roll of the robot body. Defaults to 0.0.
            pitch: Pitch of the robot body. Defaults to 0.0.
            height: Height of the robot body from normal stand height. Defaults to 0.0.
        """

        orientation = EulerZXY(yaw=yaw, roll=roll, pitch=pitch)
        cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.0,
                                                        footprint_R_body=orientation)
        self._start_robot_command('body_pitch', cmd)

    def _arm_cylindrical_velocity_cmd_helper_LOCK(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """ Helper function to build a arm velocity command from unitless cylindrical coordinates.

        params:
        + desc: string description of the desired command
        + v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        + v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        + v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        """
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)


    def _arm_angular_velocity_cmd_helper_LOCK(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """ Helper function to build a arm velocity command from angular velocities measured with respect
            to the odom frame, expressed in the hand frame.

        params:
        + desc: string description of the desired command
        + v_rx: angular velocity about X-axis in units rad/sec
        + v_ry: angular velocity about Y-axis in units rad/sec
        + v_rz: angular velocity about Z-axis in units rad/sec

        """
        # Specify a zero linear velocity of the hand. This can either be in a cylindrical or Cartesian coordinate system.
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)


    def _arm_angular_velocity_cmd_helper(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """ Helper function to build a arm velocity command from angular velocities measured with respect
            to the odom frame, expressed in the hand frame.

        params:
        + desc: string description of the desired command
        + v_rx: angular velocity about X-axis in units rad/sec
        + v_ry: angular velocity about Y-axis in units rad/sec
        + v_rz: angular velocity about Z-axis in units rad/sec

        """
        # Specify a zero linear velocity of the hand. This can either be in a cylindrical or Cartesian coordinate system.
        cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity(frame_name = "body")

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cartesian_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)


    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    def _stop(self):
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    def _arm_cartesian_velocity_cmd_helper(self, desc, v_x=0.0, v_y=0.0, v_z=0.0):

        cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity(frame_name = "body")
        cartesian_velocity.velocity_in_frame_name.x = v_x
        cartesian_velocity.velocity_in_frame_name.y = v_y
        cartesian_velocity.velocity_in_frame_name.z = v_z
        #LOGGER.info(cartesian_velocity.frame_name)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cartesian_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)


    def _arm_cylindrical_velocity_cmd_helper(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """ Helper function to build a arm velocity command from unitless cylindrical coordinates.

        params:
        + desc: string description of the desired command
        + v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        + v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        + v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        """
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity(frame_name = "body")
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        self._start_robot_command(
            desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _arm_velocity_cmd_helper(self, arm_velocity_command, desc=''):

        # Build synchronized robot command
        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
            arm_velocity_command)

        return self._start_robot_command(desc, robot_command,
                                  end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _stow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_stow_command())
        

    def _unstow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_ready_command())


    def _toggle_power(self):
        power_state = self._power_state()
        if power_state is None:
            self.add_message('Could not toggle power because power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            self._try_grpc_async('powering-on', self._request_power_on)
        else:
            self._try_grpc('powering-off', self._safe_power_off)

    def _request_power_on(self):
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self._power_client.power_command_async(request)

    def _safe_power_off(self):
        self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())

    def _power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    def _lease_str(self, lease_keep_alive):
        if lease_keep_alive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = lease_keep_alive.lease_wallet.get_lease()
                lease = f'{_lease.lease_proto.resource}:{_lease.lease_proto.sequence}'
            except bosdyn.client.lease.Error:
                lease = '...'
            if lease_keep_alive.is_alive():
                alive = 'RUNNING'
            else:
                alive = 'STOPPED'
        return f'Lease {lease} THREAD:{alive}'

    def _power_state_str(self):
        power_state = self._power_state()
        if power_state is None:
            return ''
        state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return f'Power: {state_str[6:]}'  # get rid of STATE_ prefix

    def _estop_str(self):
        if not self._estop_client:
            thread_status = 'NOT ESTOP'
        else:
            thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
        estop_status = '??'
        state = self.robot_state
        if state:
            for estop_state in state.estop_states:
                if estop_state.type == estop_state.TYPE_SOFTWARE:
                    estop_status = estop_state.State.Name(estop_state.state)[6:]  # s/STATE_//
                    break
        return f'Estop {estop_status} (thread: {thread_status})'

    def _time_sync_str(self):
        if not self._robot.time_sync:
            return 'Time sync: (none)'
        if self._robot.time_sync.stopped:
            status = 'STOPPED'
            exception = self._robot.time_sync.thread_exception
            if exception:
                status = f'{status} Exception: {exception}'
        else:
            status = 'RUNNING'
        try:
            skew = self._robot.time_sync.get_robot_clock_skew()
            if skew:
                skew_str = f'offset={duration_str(skew)}'
            else:
                skew_str = '(Skew undetermined)'
        except (TimeSyncError, RpcError) as err:
            skew_str = f'({err})'
        return f'Time sync: {status} {skew_str}'

    def _battery_str(self):
        if not self.robot_state:
            return ''
        battery_state = self.robot_state.battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # get rid of STATUS_ prefix
        if battery_state.charge_percentage.value:
            bar_len = int(battery_state.charge_percentage.value) // 10
            bat_bar = f'|{"=" * bar_len}{" " * (10 - bar_len)}|'
        else:
            bat_bar = ''
        time_left = ''
        if battery_state.estimated_runtime:
            time_left = f'({secs_to_hms(battery_state.estimated_runtime.seconds)})'
        return f'Battery: {status}{bat_bar} {time_left}'

    def _setup_file_logger(self,loggerName, fileName, level=logging.INFO):
        handler = logging.FileHandler(fileName)
        log_formatter = logging.Formatter('%(message)s')#('%(created)f - [[[%(message)s]]]')
        handler.setFormatter(log_formatter)

        fileHandlerLogger = logging.getLogger(loggerName)
        fileHandlerLogger.setLevel(level)
        fileHandlerLogger.addHandler(handler)

        return fileHandlerLogger

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.grasp_position is None:
                cv2.circle(self.image, (x, y), 10, (255, 0, 0), 5)
                cv2.imshow('image', self.image)
                self.grasp_position = (x, y)
                print((x,y))


    def show_image_get_point(self,window_name='Show Captured Image'):
        """Open window showing the side by side fisheye images with on-screen prompts for user."""

        u._draw_text_on_image(self.image, 'Click handle.')
        cv2.imshow(window_name, self.image)
        cv2.setMouseCallback(window_name, self._on_mouse)

        while self.grasp_position is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print('"q" pressed, exiting.')
                cv2.destroyAllWindows()
                break
        print(
            f'Picking object at image location ({self.grasp_position[0]}, {self.grasp_position[1]})')

    def _show_image_grasp(self):
        self.grasp_position = None

        image_res = self._image_client.get_image_from_sources(self.image_source)
        for response in image_res:
            if response.source.image_type == ImageSource.IMAGE_TYPE_VISUAL:
                # Convert image proto to CV2 image, for display later.
                image = np.frombuffer(response.shot.image.data, dtype=np.uint8)
                image = cv2.imdecode(image, -1)
                self.image = image
                self.image_rep = response
            else:
                self.depth_image = response.shot.image.data
        self.show_image_get_point()
        pick_vec = geometry_pb2.Vec2(x=self.grasp_position[0], y=self.grasp_position[1])
        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=self.image_rep.shot.transforms_snapshot,
            frame_name_image_sensor=self.image_rep.shot.frame_name_image_sensor,
            camera_model=self.image_rep.source.pinhole)
        u.add_grasp_constraint(4, grasp, self._robot_state_client)
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = self._robot_mani_client.manipulation_api_command(
            manipulation_api_request=grasp_request)
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = self._robot_mani_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(
                f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}'
            )

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)



    #in body frame
    def _grasp_from_agent(self,):
        self.grasp_position = None
        image_res = self._image_client.get_image_from_sources(self.image_source)
        for response in image_res:
            if response.source.image_type == ImageSource.IMAGE_TYPE_VISUAL:
                # Convert image proto to CV2 image, for display later.
                image = np.frombuffer(response.shot.image.data, dtype=np.uint8)
                image = cv2.imdecode(image, -1)
                self.image = image
            else:
                self.depth_image = response.shot.image.data
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        image_ten = torch.from_numpy(rgb_image)
        feature = self.imgprocessor.encoder.feature_extract_single(image_ten)
        mask = None
        if afford:
            self.show_image_get_point()
            centroid=self.grasp_position
            LOGGER.info(centroid)
            mask = self.imgprocessor.segmentor.get_segment_single_promt(image_ten,centroid)



        # # need refine

        obs={
            'policy':feature,
            'mask':mask
        }
        action_x, action_y  =agent.select_action(obs)
        self.grasp_from_image(x=action_x,y=action_y)
        LOGGER.info(f'action_x is{action_x,action_y}')


    def grasp_from_image(self,x=86,y=309):
        image_res = self._image_client.get_image_from_sources(self.image_source)
        for response in image_res:
            if response.source.image_type == ImageSource.IMAGE_TYPE_VISUAL:
                # Convert image proto to CV2 image, for display later.
                image = np.frombuffer(response.shot.image.data, dtype=np.uint8)
                image = cv2.imdecode(image, -1)
                self.image = image
                self.image_rep = response
        pick_vec = geometry_pb2.Vec2(x=x, y=y)
        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=self.image_rep.shot.transforms_snapshot,
            frame_name_image_sensor=self.image_rep.shot.frame_name_image_sensor,
            camera_model=self.image_rep.source.pinhole)
        u.add_grasp_constraint(2, grasp, self._robot_state_client)
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = self._robot_mani_client.manipulation_api_command(
            manipulation_api_request=grasp_request)
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = self._robot_mani_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            LOGGER.info(
                f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}'
            )

            if response.current_state != manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break




def _setup_logging(verbose):
    """Log to file at debug level, and log to console at INFO or DEBUG (if verbose).

    Returns the stream/console logger so that it can be removed when in curses mode.
    """
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Save log messages to file wasd.log for later debugging.
    file_handler = logging.FileHandler('control.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    # The stream handler is useful before and after the application is in curses-mode.
    if verbose:
        stream_level = logging.DEBUG
    else:
        stream_level = logging.INFO

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(log_formatter)
    LOGGER.addHandler(stream_handler)
    return stream_handler



def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--time-sync-interval-sec',
                        help='The interval (seconds) that time-sync estimate should be updated.',
                        type=float)
    options = parser.parse_args()
    stream_handler = _setup_logging(options.verbose)



    # Create robot object.
    sdk = create_standard_sdk('WASDClient')
    robot = sdk.create_robot('10.0.0.30')
    try:
        robot.authenticate('rllab', 'robotlearninglab')
        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync(options.time_sync_interval_sec)
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
        return False
    
    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    wasd_interface = WasdInterface(robot)

    try:
        wasd_interface.start()
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed to initialize robot communication: %s', err)
        return False

    LOGGER.removeHandler(stream_handler)  # Don't use stream handler in curses mode.

    try:
        try:
            # Prevent curses from introducing a 1 second delay for ESC key
            os.environ.setdefault('ESCDELAY', '0')
            # Run wasd interface in curses mode, then restore terminal config.
            curses.wrapper(wasd_interface.drive)
        finally:
            # Restore stream handler to show any exceptions or final messages.
            LOGGER.error('Failed to initialize robot communication')
    except Exception as e:
        LOGGER.error('WASD has thrown an error: [%r] %s', e, e, exc_info=True)
    finally:
        # Do any final cleanup steps.
        wasd_interface.shutdown()

    return True

if __name__ == '__main__':
    if not main():
        os._exit(1)
    os._exit(0)
