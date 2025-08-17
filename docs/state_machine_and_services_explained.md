# State Machine and ROS Services Explained

## Part 1: What are ROS Services?

### Topics vs Services Comparison

| Aspect | Topics | Services |
|--------|--------|----------|
| **Pattern** | Publish-Subscribe | Request-Response |
| **Communication** | One-to-many | One-to-one |
| **Feedback** | No confirmation | Gets response |
| **Use case** | Streaming data (sensors, telemetry) | Commands, queries |
| **Example** | Publishing velocity commands | Arming the drone |

### Service Structure

```
Client (asks)                    Server (responds)
     │                                │
     ├──────── REQUEST ──────────────>│
     │         (input data)           │
     │                                │ (processes request)
     │                                │
     │<──────── RESPONSE ─────────────┤
               (success + result)
```

### Real Service Example: Arming the Drone

```python
# SERVICE DEFINITION (CommandBool.srv):
bool value    # Request: true to arm, false to disarm
---
bool success  # Response: did it work?
uint8 result  # Response: error code if failed

# CLIENT CODE (your state machine):
from mavros_msgs.srv import CommandBool

# Create service client
arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

# Call the service
try:
    response = arming_client(True)  # Request: arm the drone
    if response.success:
        print("Drone armed!")
    else:
        print("Arming failed!")
except rospy.ServiceException as e:
    print(f"Service call failed: {e}")
```

---

## Part 2: What Does Our State Machine Do?

### The State Machine's Job

The state machine is like a **Mission Director** that:
1. Keeps track of what the drone should be doing (states)
2. Decides when to switch between behaviors (transitions)
3. Coordinates all the subsystems (via services)

### The States (What the drone can be doing)

```
┌─────────────┐
│    IDLE     │ (Waiting to start)
└──────┬──────┘
       │ start_mission
┌──────▼──────┐
│GPS_NAVIGATION│ (Flying to GPS waypoint)
└──────┬──────┘
       │ visual_target_acquired
┌──────▼──────┐
│VISUAL_SERVOING│ (Approaching ArUco marker)
└──────┬──────┘
       │ visual_goal_reached
┌──────▼──────┐
│PERFORMING_TASK│ (Injecting the nest)
└──────┬──────┘
       │ task_completed
┌──────▼──────┐
│RETURNING_HOME│ (Flying back)
└──────┬──────┘
       │ home_reached
┌──────▼──────┐
│   LANDING   │ (Mission complete)
└─────────────┘
```

### State Transitions (When to switch)

```python
# Transition example:
self.add_transition(
    trigger='visual_target_acquired',     # Event name
    source='gps_navigation',              # Current state
    dest='visual_servoing',               # Next state
    conditions=['is_aruco_detected_confident', 'is_within_visual_range'],  # Guards
    before='save_gps_position',           # Action before transition
    after='enable_visual_mode'            # Action after transition
)
```

---

## Part 3: How State Machine Uses Services

### Old Way (Launching/Killing Nodes) ❌
```python
def on_enter_gps_navigation(self):
    # Kill previous nodes
    self.manager.stop_current_process()  
    
    # Launch new nodes
    self.manager.launch_file('package', 'gps_navigation.launch')
    
    # Problems:
    # - Slow (2-3 seconds)
    # - Unstable (nodes die/restart)
    # - Loses state information
```

### New Way (Using Services) ✅
```python
def on_enter_gps_navigation(self):
    # Call service to change navigation mode
    request = SetNavigationModeRequest()
    request.mode = "gps_tracking"
    response = self.nav_mode_service.call(request)
    
    if response.success:
        rospy.loginfo("Switched to GPS mode")
    
    # Advantages:
    # - Fast (milliseconds)
    # - Stable (nodes keep running)
    # - Maintains state/history
```

---

## Part 4: Complete Example Flow

Let's trace what happens when the drone sees an ArUco marker:

### 1. ArUco Detection
```python
# ArUco detector publishes to topic
/aruco_detection_status → True
```

### 2. State Machine Receives Event
```python
def aruco_status_callback(self, msg):
    if msg.data and self.state == "gps_navigation":
        self.visual_target_acquired()  # Trigger transition!
```

### 3. State Machine Calls Services
```python
def on_enter_visual_servoing(self):
    # Service 1: Tell Navigation Manager to switch mode
    nav_request = SetNavigationModeRequest()
    nav_request.mode = "visual_servoing"
    nav_response = self.nav_mode_service.call(nav_request)
    
    # Service 2: Ensure drone is in OFFBOARD mode
    mode_request = SetModeRequest()
    mode_request.custom_mode = "OFFBOARD"
    mode_response = self.mavros_mode_service.call(mode_request)
    
    # Service 3: Ensure drone is armed
    arm_request = CommandBoolRequest()
    arm_request.value = True
    arm_response = self.arming_service.call(arm_request)
```

### 4. Navigation Manager Responds
```python
# Inside Navigation Mode Manager (service server)
def handle_set_mode(self, request):
    if request.mode == "visual_servoing":
        self.current_mode = NavigationMode.VISUAL_SERVOING
        # Nodes automatically adjust their behavior
        # No killing/restarting!
        return SetNavigationModeResponse(
            success=True,
            message="Switched to visual servoing"
        )
```

### 5. All Nodes Adapt
```
Navigation Manager → Broadcasts new mode on topic
PID Controller → Switches to visual gains
APF Node → Uses gentler obstacle avoidance
State Machine → Gets confirmation via service response
```

---

## Services Used in Our System

### 1. Navigation Mode Service (Custom)
```python
# SetNavigationMode.srv
string mode  # "gps_tracking", "visual_servoing", etc.
---
bool success
string message
```

### 2. MAVROS Services (Built-in)
```python
# Arming
/mavros/cmd/arming → CommandBool

# Flight Mode
/mavros/set_mode → SetMode

# Takeoff
/mavros/cmd/takeoff → CommandTOL

# Land
/mavros/cmd/land → CommandTOL
```

---

## Why Services are Better for State Machines

### 1. **Confirmation**
```python
# With topics: 
cmd_pub.publish(command)  # Did it work? 🤷

# With services:
response = service.call(request)
if not response.success:
    self.handle_failure()  # We KNOW it failed!
```

### 2. **Synchronization**
```python
# Services wait for completion
response = arm_service.call(True)
# Drone is now armed (or we know it failed)

# Topics don't wait
arm_pub.publish(True)
# Is it armed yet? Who knows!
```

### 3. **Error Handling**
```python
try:
    response = set_mode_service.call("OFFBOARD")
    if not response.success:
        # Try alternative approach
        self.fallback_mode()
except rospy.ServiceException:
    # Service is down, enter emergency mode
    self.emergency_land()
```

---

## Quick Service Commands

### Create a Service
```bash
# 1. Define service in srv/MyService.srv
string input
---
bool success

# 2. Add to CMakeLists.txt
add_service_files(FILES MyService.srv)

# 3. Build
catkin_make
```

### Call a Service from Terminal
```bash
# See available services
rosservice list

# Check service type
rosservice type /mavros/cmd/arming

# Call service manually
rosservice call /mavros/cmd/arming "value: true"

# Call custom navigation service
rosservice call /navigation/set_mode "mode: 'visual_servoing'"
```

### Monitor Services
```bash
# See if service exists
rosservice info /navigation/set_mode

# Watch service calls happen
rosservice echo /navigation/set_mode
```

---

## Summary

**State Machine**: The mission director that coordinates everything
- Tracks mission progress
- Makes decisions about what to do next
- Ensures smooth transitions

**Services**: Request-response communication for critical commands
- Get confirmation that commands worked
- Handle errors gracefully
- Perfect for state changes

**Together**: State machine uses services to reliably control the system
- No more launching/killing nodes
- Fast, stable mode changes
- Proper error handling

The state machine is like a conductor of an orchestra, and services are how it tells each section when to play!