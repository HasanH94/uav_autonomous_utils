# Control Modes Explanation

## PID Controller Operation Modes

### 1. **Position Setpoint Mode** (GPS Waypoints & Visual Servoing)
```
Input: Target Position (x, y, z) + Target Yaw
       ↓
Error: e = target_pos - current_pos
       ↓
PID:   u = Kp*e + Ki*∫e + Kd*ė
       ↓
Output: Velocity Command (vx, vy, vz, ωz)
```

**Characteristics:**
- Single goal position
- Full PID control (P + I + D)
- Integral term helps eliminate steady-state error
- Works well for hovering at a point

**Use Cases:**
- GPS waypoint navigation
- Visual servoing to ArUco marker
- Hovering at injection position

---

### 2. **Trajectory Tracking Mode** (Following Planned Paths)
```
Input: Trajectory (sequence of poses + optional velocities)
       ↓
Carrot: Find look-ahead point on trajectory
       ↓
Error: e = carrot_pos - current_pos
       ↓
PD+FF: u = Kp*e + Kd*ė + Kff*trajectory_velocity
       ↓
Output: Velocity Command (vx, vy, vz, ωz)
```

**Characteristics:**
- Following a moving reference point (carrot)
- Usually PD control (no integral)
- Feedforward from trajectory velocities
- Look-ahead prevents oscillations

**Use Cases:**
- Following global planner output
- Smooth path through obstacles
- Energy-optimal trajectories

---

## Key Differences

| Aspect | Position Setpoint | Trajectory Tracking |
|--------|------------------|-------------------|
| **Input** | Single goal pose | Sequence of poses/velocities |
| **Reference** | Fixed target | Moving carrot point |
| **Error Calculation** | target - current | carrot - current |
| **Control Type** | PID (with integral) | PD + Feedforward |
| **Integral Term** | Yes (eliminates offset) | No (would cause lag) |
| **Look-ahead** | Not needed | Critical for smoothness |
| **Best For** | Reaching & holding position | Following smooth paths |

---

## Example Scenarios

### Scenario 1: GPS Navigation to Nest
```
Mode: Position Setpoint
1. Receive GPS waypoint (12, 5, 10)
2. Calculate error: (12, 5, 10) - current_pos
3. Apply PID control
4. Output velocity to reduce error
```

### Scenario 2: Visual Servoing
```
Mode: Position Setpoint
1. Receive visual target with exact orientation
2. Calculate position AND orientation error
3. Apply PID control (tighter gains)
4. Output velocity for precise alignment
```

### Scenario 3: Following Global Planner Path
```
Mode: Trajectory Tracking
1. Receive trajectory with 100 points
2. Find carrot point 3m ahead on path
3. Calculate error to carrot (not end goal!)
4. Apply PD control + feedforward
5. Smoothly follow the path
```

---

## Why Different Gains?

### Position Setpoint Gains
- **Higher Ki**: Need integral to eliminate steady-state error
- **Moderate Kp**: Prevent overshoot at target
- **Lower Kd**: Less damping needed

### Trajectory Tracking Gains
- **Higher Kp**: Quick response to track moving carrot
- **Zero Ki**: Integral would cause trajectory lag
- **Higher Kd**: More damping for smooth following
- **Feedforward**: Anticipate trajectory velocities

---

## Implementation in Our System

```yaml
# Position mode (GPS/Visual)
/move_base_gps → Position Setpoint Mode → PID Control
/move_base_visual → Position Setpoint Mode → PID Control

# Trajectory mode (Global Planner)
/global_planner/trajectory → Trajectory Tracking Mode → PD+FF Control
```

The same controller handles both by switching internal algorithms based on input type!