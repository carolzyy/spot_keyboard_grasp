from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.lease import LeaseClient
import bosdyn
# Connect to Spot
sdk = create_standard_sdk('spot_lease_checker')
robot = sdk.create_robot('10.0.0.30')  # Replace with Spot's IP
robot.authenticate('rllab', 'robotlearninglab')
bosdyn.client.util.authenticate(robot)

# Create lease client
lease_client = robot.ensure_client(LeaseClient.default_service_name)

# List active leases
#active_leases = lease_client.list_leases()
#print("Active Leases:", active_leases)


lease = lease_client.take()  # Or use lease_client.take() to forcefully claim it
print(f"Acquired Lease: {lease}")
lease_client.return_lease(lease)
print("Lease returned.")