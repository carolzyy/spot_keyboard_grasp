import argparse
import curses
import logging
import os
import signal
import threading
import time

import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncPeriodicGRPCTask
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, format_metric, secs_to_hms
from bosdyn.geometry import EulerZXY
import pickle
import curses
import io
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import OrderedDict

from PIL import Image, ImageEnhance

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.power_pb2 as PowerServiceProto
# import bosdyn.api.robot_command_pb2 as robot_command_pb2
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.api.spot.robot_command_pb2 as spot_command_pb2
import bosdyn.client.util
from bosdyn.api import geometry_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME,VISION_FRAME_NAME,BODY_FRAME_NAME,HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, format_metric, secs_to_hms
from bosdyn.client import math_helpers
import argparse
import sys
import time

from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, estop_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand, blocking_sit)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import duration_to_seconds

import traceback
from datetime import datetime
import os, errno

import numpy as np   

###############
#from offline_planner import *

LOGGER = logging.getLogger(__name__)


# Logs name
# datetime object containing current date and time
# dd/mm/YY H:M:S
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
ACTION_LOG_PATH='logs-wall/'+dt_string+'/'
STATE_LOG_PATH='logs-wall/'+dt_string+'/'
ARM_LOG_PATH='logs-wall/'+dt_string+'/arms/' 
IMG_LOG_PATH='logs-wall/'+dt_string+'/image/'


os.makedirs(ACTION_LOG_PATH)
os.makedirs(IMG_LOG_PATH, exist_ok=True)
os.makedirs(ARM_LOG_PATH, exist_ok=True)


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

def _image_to_ascii(image, new_width):
    """Convert an rgb image to an ASCII 'image' that can be displayed in a terminal."""

    ASCII_CHARS = '@#S%?*+;:,.'

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(0.8)

    # Scaling image before rotation by 90 deg.
    scaled_rot_height = new_width
    original_rot_width, original_rot_height = image.size
    scaled_rot_width = (original_rot_width * scaled_rot_height) // original_rot_height
    # Scaling rotated width (height, after rotation) by half because ASCII chars
    #  in terminal seem about 2x as tall as wide.
    image = image.resize((scaled_rot_width // 2, scaled_rot_height))

    # Rotate image 90 degrees, then convert to grayscale.
    image = image.transpose(Image.ROTATE_270)
    image = image.convert('L')

    def _pixel_char(pixel_val):
        return ASCII_CHARS[pixel_val * len(ASCII_CHARS) // 256]

    img = []
    row = [' '] * new_width
    last_col = new_width - 1
    for idx, pixel_char in enumerate(_pixel_char(val) for val in image.getdata()):
        idx_row = idx % new_width
        row[idx_row] = pixel_char
        if idx_row == last_col:
            img.append(''.join(row))
    return img

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
            arm_joint_states = []
            for joint in result.kinematic_state.joint_states:
                if 'arm' in joint.name:
                    arm_joint_states.append(joint)
            if arm_joint_states[2] ==0:
                arm_joint_states = np.delete(arm_joint_states, 2)
                print('deteleted 0')

            arm_filename = ARM_LOG_PATH + str(timestamp)+'.pkl'

            with open(arm_filename, 'wb') as f:
                pickle.dump(arm_joint_states, f)

        except Exception as e:
            LOGGER.exception('Error saving the image: %s', e)

    def _handle_error(self, exception):
        LOGGER.exception('Failure getting image: %s', exception)


class AsyncImageCapture(AsyncPeriodicGRPCTask):
    """Grab camera images from the robot."""

    def __init__(self, robot, source_name, save_mode):
        super(AsyncImageCapture, self).__init__(period_sec=0.0001)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._ascii_image = None
        self._video_mode = False
        self._should_take_image = False
        self.source_name = source_name
        self.save_mode = save_mode

    @property
    def ascii_image(self):
        """Return the latest captured image as ascii."""
        return self._ascii_image

    def toggle_video_mode(self):
        """Toggle whether doing continuous image capture."""
        self._video_mode = not self._video_mode

    def take_image(self):
        """Request a one-shot image."""
        self._should_take_image = not self._should_take_image

    def _start_query(self):
        source_name = self.source_name 
        return self._image_client.get_image_from_sources_async([source_name])

    def _should_query(self, now_sec):  # pylint: disable=unused-argument
        return (self._video_mode or self._should_take_image) and (now_sec - self._last_call) > self._period_sec

    def _handle_result(self, result):
        try:
            mydir = os.path.join(
                            IMG_LOG_PATH, self.source_name) 
            isExist = os.path.exists(mydir)
            if not isExist:
                os.makedirs(mydir)
            imageFileName = "%.20f" % time.time() #datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
            image = []
            if(self.save_mode == 1):
                image = Image.open(io.BytesIO(result[0].shot.image.data))
            else:
                image = np.frombuffer(result[0].shot.image.data, dtype=np.uint16)
                image = image.reshape(result[0].shot.image.rows,
                                            result[0].shot.image.cols)
                image = Image.fromarray(image)

            image = image.save(os.path.join(mydir,imageFileName + '.png'))

            global PREV_IMG
            PREV_IMG = image

        except Exception as e:
            LOGGER.exception('Error saving the image: %s', e)

    def _handle_error(self, exception):
        LOGGER.exception('Failure getting image: %s', exception)


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
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        #['frontright_fisheye_image', 'frontleft_fisheye_image', 'frontright_depth', 'frontleft_depth']
        self._image_task_frontright_image = AsyncImageCapture(robot, 'frontright_fisheye_image', 1) #TODO : Change to constant or do something
        self._image_task_frontleft_image = AsyncImageCapture(robot, 'frontleft_fisheye_image', 1)
        self._image_task_frontright_depth = AsyncImageCapture(robot, 'frontright_depth_in_visual_frame', 2)
        self._image_task_frontleft_depth = AsyncImageCapture(robot, 'frontleft_depth_in_visual_frame', 2)
        self._arm_log_task = AsyncArmStateCapture(self._robot_state_client)
        self._async_tasks = AsyncTasks([self._robot_state_task, 
                                        self._arm_log_task,
                                        self._image_task_frontright_image,
                                        self._image_task_frontleft_image, 
                                        self._image_task_frontright_depth, 
                                        self._image_task_frontleft_depth])
        self._lock = threading.Lock()
        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            ord('x'): self._toggle_time_sync,
            ord(' '): self._toggle_estop,
            ord('r'): self._self_right,
            ord('P'): self._toggle_power,
            ord('p'): self._toggle_power,
            ord('c'): self._sit,
            ord('b'): self._battery_change_pose,
            ord('f'): self._stand,
            ord('w'): self._move_forward,
            ord('s'): self._move_backward,
            ord('a'): self._strafe_left,
            ord('d'): self._strafe_right,
            ord('W'): self._arm_cartesian_move_out,
            ord('S'): self._arm_cartesian_move_in,
            ord('A'): self._rotate_cartesian_ccw,
            ord('D'): self._rotate_cartesian_cw,
            ord('R'): self._move_up,
            ord('F'): self._move_down,
            ord('I'): self._rotate_plus_ry,
            ord('K'): self._rotate_minus_ry,
            ord('U'): self._rotate_plus_rx,
            ord('O'): self._rotate_minus_rx,
            ord('J'): self._rotate_plus_rz,
            ord('L'): self._rotate_minus_rz,
            ord('N'): self._toggle_gripper_open,
            ord('M'): self._toggle_gripper_closed,
            ord('q'): self._turn_left,
            ord('e'): self._turn_right,
            ord('i'): self._toggle_image_capture,
            ord('y'): self._unstow,
            ord('h'): self._stow,
            ord('l'): self._toggle_lease,
            ord('G'): self._body_pitch_down,
            ord('T'): self._body_pitch_up,
            ord('V'): self._reset_height,
            ord('1'): self._move_arm_position,# self._body_yaw_left,
            ord('2'): self._body_yaw_right,
            ord('9'): self.reset_LOCK,
            ord('0'): self.reset_UNLOCK,
            ord('.'): self.reset_AI_CONTROL, 
            ord(','): self.reset_HUMAN_CONTROL
            
        }
        self._locked_messages = ['', '', '']  # string: displayed message for user
        self._estop_keepalive = None
        self._exit_check = None

        # Stuff that is set in start()
        self._robot_id = None
        self._lease_keepalive = None

        self.action_handler_logger = self._setup_file_logger(loggerName="action_logger",fileName=ACTION_LOG_PATH+"action_log.log")
        self.robot_state_handler_logger = self._setup_file_logger(loggerName="robot_state_logger",fileName=STATE_LOG_PATH+"robot_state_log.log")

        # initialize the offline planner
        #self.planner = OfflinePlanner()
        self.ai_image_client = robot.ensure_client(ImageClient.default_service_name)


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
                            # Get the robot state
                            '''
                            robot_state = state_client.get_robot_state()

                            # If the robot has an arm, the arm state will be present.
                            # Extract the arm joint states if available.
                            if robot_state.HasField('arm_state'):
                                arm_joint_states = robot_state.arm_state.joint_states
                                for joint_state in arm_joint_states:
                                    print(f"Joint {joint_state.name}: Position = {joint_state.position}, Velocity = {joint_state.velocity}")
                            else:
                                print("Robot does not have an arm or the arm state is not available.")
                            '''

                            state = self._robot_state_client.get_robot_state()
                            arm = []
                            for joint_state in state.kinematic_state.joint_states:
                                if 'arm' in joint_state.name:
                                    arm.append(joint_state.position.value)

                            cam_names = ['frontleft_fisheye_image', 'frontright_fisheye_image']
                            im_res = self.ai_image_client.get_image_from_sources(cam_names)
                            image_l = Image.open(io.BytesIO(im_res[0].shot.image.data)).convert('RGB')
                            image_r = Image.open(io.BytesIO(im_res[1].shot.image.data)).convert('RGB')

                            cmds, _ = [],[]#self.planner.plan_pred([image_l, image_r], arm)
                            #AI_CONTROL = ai_control
                            #print(cmds)
                            for cm in cmds[:2]:
                                # Do not queue up commands on client
                                #print(cm)
                                for i in range(2):
                                    #print(cm)
                                    self.flush_and_estop_buffer(stdscr)
                                    self._drive_cmd(ord(cm))
                                    time.sleep(COMMAND_INPUT_RATE)
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
        stdscr.addstr(10, 0, 'Commands: [TAB]: quit                                  ')
        stdscr.addstr(11, 0, '          [T]: Time-sync, [SPACE]: Estop, [P]: Power   ')
        stdscr.addstr(12, 0, '          [i]: Take image , [O]: Video mode N/A            ')
        stdscr.addstr(13, 0, '          [f]: Stand, [r]: Self-right                  ')
        stdscr.addstr(14, 0, '          [c]: Sit, [b]: Battery-change                ')
        stdscr.addstr(15, 0, '          [y]: Unstow arm, [h]: Stow arm               ')
        stdscr.addstr(16, 0, '          [wasd]: Directional strafing                 ')
        stdscr.addstr(17, 0, '          [WASD]: Arm Radial/Azimuthal control         ')
        stdscr.addstr(18, 0, '          [RF]: Up/Down control         ')
        stdscr.addstr(19, 0, '          [UO]: X-axis rotation control             ')
        stdscr.addstr(20, 0, '          [IK]: Y-axis rotation control             ')
        stdscr.addstr(21, 0, '          [JL]: Z-axis rotation control             ')
        stdscr.addstr(22, 0, '          [NM]: Open/Close gripper                  ')
        stdscr.addstr(23, 0, '          [qe]: Body Turning, [ESC]: Stop                   ') 
        stdscr.addstr(24, 0, '          [l]: Return/Acquire lease                  ')
        stdscr.addstr(25, 0, '')   

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


        # print as many lines of the image as will fit on the curses screen
        '''if self._image_task.ascii_image != None:
                                    max_y, _max_x = stdscr.getmaxyx()
                                    for y_i, img_line in enumerate(self._image_task.ascii_image):
                                        if y_i + 17 >= max_y:
                                            break
                        
                                        stdscr.addstr(y_i + 17, 0, img_line)'''

        stdscr.refresh()

    def _drive_cmd(self, key):
        """Run user commands at each update."""
        try:
            cmd_function = self._command_dictionary[key]
            if(key not in [ord(' '), ord('\t')]):
                act_info = "%.20f" % time.time() + '-' + chr(key) 
                self.action_handler_logger.info(act_info)
            cmd_function()

        except KeyError:
            if key and key != -1 and key < 256:
                self.add_message(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _try_grpc(self, desc, thunk):
        try:
            cmd_id = thunk()
            self.robot_state_handler_logger.info(self.robot_state)
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

    def _toggle_time_sync(self):
        if self._robot.time_sync.stopped:
            self._robot.start_time_sync()
        else:
            self._robot.time_sync.stop()

    def _toggle_estop(self):
        """toggle estop on/off. Initial state is ON"""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    def _toggle_image_capture(self):
        self._image_task_frontright_image.take_image()
        self._image_task_frontleft_image.take_image()
        self._image_task_frontright_depth.take_image()
        self._image_task_frontleft_depth.take_image()

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

    def _self_right(self):
        self._start_robot_command('self_right', RobotCommandBuilder.selfright_command())

    def _battery_change_pose(self):
        # Default HINT_RIGHT, maybe add option to choose direction?
        self._start_robot_command(
            'battery_change_pose',
            RobotCommandBuilder.battery_change_pose_command(
                dir_hint=basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT))

    def _sit(self):
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())
        #self._start_robot_command('open_gripper', RobotCommandBuilder.claw_gripper_open_command())
        self._toggle_image_capture()
        self._move_arm_position()


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
            #do something
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rx(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)
        else:
            #do something
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_ry(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)
        else:
            #do something
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_ry(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)
        else:
            #do something
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_rz(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)
        else:
            #do something
            self._arm_angular_velocity_cmd_helper_LOCK('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rz(self):
        if UNLOCK:
            self._arm_angular_velocity_cmd_helper('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)
        else:
            #do something
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

    def _body_pitch_up(self):
        self._orientation_cmd_helper(pitch=0.1)

    def _body_pitch_down(self):
        self._orientation_cmd_helper(pitch=-0.7)

    def _body_yaw_left(self):
        self._orientation_cmd_helper(yaw=0.1)

    def _body_yaw_right(self):
        self._orientation_cmd_helper(yaw=-0.1)


    def reset_AI_CONTROL(self):
        global AI_CONTROL
        AI_CONTROL = True
        print('now AI control')

    def reset_HUMAN_CONTROL(self):
        global AI_CONTROL
        AI_CONTROL = False

        print('now Human control')

    def reset_UNLOCK(self):
        global UNLOCK
        UNLOCK = True

    def reset_LOCK(self):
        global UNLOCK
        UNLOCK = False

    def _reset_height(self):
        """Resets robot body height to normal stand height.
        """
        self._orientation_cmd_helper(height=0.0)


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
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity(frame_name = "body")

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)
       

    def make_robot_command(self,arm_joint_traj):
        """ Helper function to create a RobotCommand from an ArmJointTrajectory.
            The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
            filled out to follow the passed in trajectory. """

        joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
        sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
        arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
        return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

    def _move_by_joint_angle(self):
        # First point position
        sh0 = -1.5
        sh1 = -0.8
        el0 = 1.7
        el1 = 0.0
        wr0 = 0.5
        wr1 = 0.0

        # First point time (seconds)
        first_point_t = 2.0

        # Build the proto for the trajectory point.
        traj_point1 = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, first_point_t)

        # Second point position
        sh0 = 1.0
        sh1 = -0.2
        el0 = 1.3
        el1 = -1.3
        wr0 = -1.5
        wr1 = 1.5

        # Second point time (seconds)
        second_point_t = 4.0

        # Build the proto for the second trajectory point.
        traj_point2 = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, second_point_t)

        # Optionally, set the maximum allowable velocity in rad/s that a joint is allowed to
        # travel at. Also set the maximum allowable acceleration in rad/s^2 that a joint is
        # allowed to accelerate at. If these values are not set, a default will be used on
        # the robot.
        # Note that if these values are set too low and the trajectories are too aggressive
        # in terms of time, the desired joint angles will not be hit at the specified time.
        # Some other ways to help the robot actually hit the specified trajectory is to
        # increase the time between trajectory points, or to only specify joint position
        # goals without specifying velocity goals.
        max_vel = wrappers_pb2.DoubleValue(value=2.5)
        max_acc = wrappers_pb2.DoubleValue(value=15)

        # Build up a proto.
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point1, traj_point2],
                                                            maximum_velocity=max_vel,
                                                            maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = self.make_robot_command(arm_joint_traj)

        # Send the request
        self._start_robot_command('move_arm_by_joint_angle', command)
        

    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    def _stop(self):
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    def _arm_cartesian_velocity_cmd_helper(self, desc, v_x=0.0, v_y=0.0, v_z=0.0):

        cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity()
        cartesian_velocity.frame_name = "body"
        cartesian_velocity.velocity_in_frame_name.x = v_x
        cartesian_velocity.velocity_in_frame_name.y = v_y
        cartesian_velocity.velocity_in_frame_name.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cartesian_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        cmd_id = self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)


    def _arm_cylindrical_velocity_cmd_helper(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
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

        cmd_id = self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

        block_until_arm_arrives(self._robot_command_client, cmd_id, 2.0)

        state = self._robot_state_client.get_robot_state()
        joint_state_dict = {}
        for joint_state in state.kinematic_state.joint_states:
            joint_state_dict[joint_state.name] = joint_state

        # Build the proto for the trajectory point.
        traj_point1 = RobotCommandBuilder.create_arm_joint_trajectory_point(
            joint_state_dict['arm0.sh0'].position.value,joint_state_dict['arm0.sh1'].position.value
            ,joint_state_dict['arm0.el0'].position.value,joint_state_dict['arm0.el1'].position.value
            ,joint_state_dict['arm0.wr0'].position.value,joint_state_dict['arm0.wr1'].position.value, 1.0)

        max_vel = wrappers_pb2.DoubleValue(value=2.5)
        max_acc = wrappers_pb2.DoubleValue(value=15)

        # Build up a proto.
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point1],
                                                            maximum_velocity=max_vel,
                                                            maximum_acceleration=max_acc)
        # Make a RobotCommand
        command = self.make_robot_command(arm_joint_traj)

        # Send the request
        #self._start_robot_command('move_arm_by_joint_angle', command)
        

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
        #self._start_robot_command('open_gripper', RobotCommandBuilder.claw_gripper_open_command())
        self._start_robot_command('stow', RobotCommandBuilder.arm_stow_command())
        

    def _unstow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_ready_command())

    def _return_to_origin(self):
        self._start_robot_command(
            'fwd_and_rotate',
            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=0.0, goal_y=0.0, goal_heading=0.0, frame_name=ODOM_FRAME_NAME, params=None,
                body_height=0.0, locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_TROT),
            end_time_secs=time.time() + 20)

    def _take_ascii_image(self):
        source_name = 'frontright_fisheye_image'
        image_response = self._image_client.get_image_from_sources([source_name])
        image = Image.open(io.BytesIO(image_response[0].shot.image.data))
        ascii_image = self._ascii_converter.convert_to_ascii(image, new_width=70)
        self._last_image_ascii = ascii_image

    def _toggle_ascii_video(self):
        if self._video_mode:
            self._video_mode = False
        else:
            self._video_mode = True


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
    def _move_arm_position(self,x=0.3,y=0,z=-0.25,qw=1,qx=0,qy=0,qz=0):
        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self._robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         BODY_FRAME_NAME, HAND_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, BODY_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        #gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(arm_command)
        #robot_command.synchronized_command.arm_command
        # Send the request
        self._start_robot_command('move_arm',command)

def _setup_logging(verbose):
    """Log to file at debug level, and log to console at INFO or DEBUG (if verbose).

    Returns the stream/console logger so that it can be removed when in curses mode.
    """
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Save log messages to file wasd.log for later debugging.
    file_handler = logging.FileHandler('customized_remote_control.log')
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
    robot = sdk.create_robot(options.hostname)
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
            LOGGER.addHandler(stream_handler)
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
