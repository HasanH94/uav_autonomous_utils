# PID Controller Conflict Resolution

## The Problem
What happens when the PID controller receives multiple, potentially conflicting inputs simultaneously?
- GPS waypoint says "go to position A"
- Visual system says "go to position B" 
- Global planner says "follow this trajectory"

## The Solution: Priority-Based Input Management

### Input Priority Hierarchy (Highest to Lowest)

1. **EMERGENCY (Priority 0)** - Safety override
   - Always accepted immediately
   - Used for collision avoidance or emergency stop

2. **VISUAL (Priority 1)** - Visual servoing for injection
   - High priority for precise nest approach
   - Overrides GPS and trajectory when ArUco detected

3. **TRAJECTORY (Priority 2)** - Global planner paths
   - Medium priority for efficient navigation
   - Can be overridden by visual when target detected

4. **GPS (Priority 3)** - Basic waypoint navigation
   - Lowest priority
   - Default fallback mode

### Conflict Resolution Rules

```python
def should_accept_input(new_input):
    # Rule 1: Always accept emergency
    if new_input == EMERGENCY:
        return ACCEPT
    
    # Rule 2: Check if current input is stale (>2 seconds old)
    if current_input_age > timeout:
        return ACCEPT  # Stale input, accept new one
    
    # Rule 3: Higher priority can override lower
    if new_priority < current_priority:  # Lower number = higher priority
        return ACCEPT
    
    # Rule 4: Same priority updates the target
    if new_priority == current_priority:
        return ACCEPT  # Update target
    
    # Rule 5: Lower priority is rejected
    return REJECT
```

## Example Scenarios

### Scenario 1: GPS + Visual Conflict
```
Time 0.0s: GPS waypoint received → Accepted (no active input)
Time 1.0s: Flying toward GPS waypoint
Time 2.0s: Visual target detected → Accepted (VISUAL > GPS)
Time 2.1s: GPS waypoint received → REJECTED (GPS < VISUAL)
Result: Continues visual servoing, ignores GPS
```

### Scenario 2: Trajectory + Visual Conflict
```
Time 0.0s: Trajectory received → Accepted
Time 5.0s: Following trajectory smoothly
Time 6.0s: Visual target detected → Accepted (VISUAL > TRAJECTORY)
Time 8.0s: Lost visual → Visual becomes stale after 2s
Time 10.1s: Trajectory received → Accepted (previous input stale)
Result: Smooth transition back to trajectory following
```

### Scenario 3: Multiple GPS Inputs
```
Time 0.0s: GPS waypoint A received → Accepted
Time 0.5s: GPS waypoint B received → Accepted (same priority, updates target)
Result: Smoothly changes target from A to B
```

### Scenario 4: Stale Input Timeout
```
Time 0.0s: Visual target received → Accepted
Time 1.0s: Visual servoing active
Time 2.0s: Visual target lost (no new updates)
Time 4.1s: GPS waypoint received → Accepted (visual is stale > 2s)
Result: Automatically falls back to GPS after timeout
```

## Configuration Parameters

```yaml
# In launch file or config
priority_mode: 'auto'        # Automatic priority handling
input_timeout: 2.0           # Seconds before input considered stale  
allow_override: true         # Allow higher priority to override
```

## Monitoring Conflicts

```bash
# See current priority
rostopic echo /pid/current_priority

# Monitor conflicts (shows rejected inputs)
rostopic echo /pid/input_conflict

# Watch mode changes
rostopic echo /pid/control_mode
```

## Benefits

1. **No Manual Switching** - Automatic priority handling
2. **Smooth Transitions** - No abrupt changes when switching inputs
3. **Failsafe Behavior** - Falls back to lower priority if higher fails
4. **Predictable** - Clear rules for what takes precedence
5. **Observable** - Can monitor what's happening via topics

## Integration with Navigation Mode Manager

The Navigation Mode Manager and PID Controller work together:

```
Navigation Mode Manager → Sets high-level mode (GPS/Visual/Search)
                       ↓
PID Controller → Handles actual inputs with priority
```

- Mode manager suggests the mode
- PID controller enforces priority when conflicts arise
- Visual always wins when close to target (for precision)
- Stale inputs automatically timeout

This ensures smooth, predictable behavior even when multiple systems are sending commands!