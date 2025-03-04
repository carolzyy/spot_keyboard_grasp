
## Setup Dependencies

See requirements.txt for the dependencies required for this example. Using pip, these dependencies can be installed using:

```
python3 -m pip install -r requirements.txt
```

## Running the Example

```
python3 keyboard_grasp_from_image.py ROBOT_IP
```

When run, this example will create an interface in your terminal listing the controls which are as follows:

| Button | Functionality                   |
|--------|---------------------------------|
| wasd   | Directional Strafing            |
| qe     | Turning                         |
| f      | Stand                           |
| c      | Sit                             |
| v      | Show image and wait for input   |
| SPACE  | E-Stop                          |
| P/p    | Motor power & Control           |
| Tab    | Exit                            |


## Troubleshooting
If you cannot run the script properly due to a Lease Error, it may be caused by an improper exit of the script. Try running:

```
python3 lease.py
```