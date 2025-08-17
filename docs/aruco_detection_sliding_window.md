# ArUco Detection Sliding Window Algorithm

## The Problem
We need to avoid false positives from single-frame detections while being robust to occasional missed frames due to:
- Motion blur
- Temporary occlusions
- Lighting changes
- Camera noise

## The Solution: Sliding Window with Tolerance

### How It Works

```
Time →
Frame:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
Detect: 0  0  1  1  1  0  1  1  1  1  0  1  1  1  0  1  1  1  1  1  0  1  1  1  1  0  1  1  1  1
Window: [--------------------------- 30 frames (~1.5 sec) ---------------------------]

Detection Rate: 23/30 = 76.7% ✓ (> 70% threshold)
Result: CONSISTENT DETECTION - Switch to Visual Servoing
```

### Configuration Parameters

```yaml
# In launch file or parameters
aruco_detection_window_size: 30    # frames (1.5 sec at 20Hz)
aruco_detection_threshold: 0.7     # 70% of frames must detect

# This means:
# - Need 21 out of 30 frames with detection
# - Can tolerate up to 9 missed frames in the window
# - Robust to temporary occlusions
```

### Algorithm

```python
# Sliding window implementation
detection_history = []  # Boolean array

for each new frame:
    # Add new detection result
    detection_history.append(aruco_detected)
    
    # Maintain window size
    if len(detection_history) > window_size:
        detection_history.pop(0)  # Remove oldest
    
    # Calculate detection rate
    if len(detection_history) == window_size:
        detection_rate = sum(detection_history) / window_size
        
        if detection_rate >= threshold:
            # Consistent detection!
            trigger_visual_servoing()
```

## Examples

### Example 1: Good Detection (Switch to Visual)
```
Frames: [1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1]
Rate: 24/30 = 80% ✓
Action: Switch to visual servoing
```

### Example 2: Sporadic Detection (Stay in GPS)
```
Frames: [0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0]
Rate: 8/30 = 26.7% ✗
Action: Continue GPS navigation
```

### Example 3: Building Confidence
```
Time 0.0s: [0,0,0,0,0,0,0,0,0,0] → 0% (Building history: 10/30)
Time 0.5s: [0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1] → 50% (Building: 20/30)
Time 1.0s: [0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1] → 60% (Full window)
Time 1.5s: [1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1] → 73% ✓
Action: NOW switch to visual servoing
```

## State Transitions

### GPS → Visual Servoing
```
Conditions Required:
1. detection_rate >= 70% (sliding window)
2. distance < 10m (within visual range)
3. state == "gps_navigation"
```

### Visual Servoing → Search
```
Conditions Required:
1. detection_rate < 70% (lost consistency)
2. state == "visual_servoing"
```

## Benefits

1. **No False Positives**: Single frame detection won't trigger mode switch
2. **Tolerates Missed Frames**: Can lose up to 30% of frames
3. **Smooth Transitions**: Natural hysteresis prevents oscillation
4. **Configurable**: Adjust window size and threshold for your environment
5. **Observable**: Can log detection rate for debugging

## Tuning Guide

### For More Robustness (Vegetation/Obstacles)
```yaml
aruco_detection_window_size: 40    # Longer window (2 seconds)
aruco_detection_threshold: 0.6     # Lower threshold (60%)
```

### For Faster Response (Clear Environment)
```yaml
aruco_detection_window_size: 20    # Shorter window (1 second)
aruco_detection_threshold: 0.8     # Higher threshold (80%)
```

### For High-Speed Flight
```yaml
aruco_detection_window_size: 15    # Very short window (0.75 seconds)
aruco_detection_threshold: 0.65    # Moderate threshold
```

## Monitoring

```bash
# Watch detection status
rostopic echo /aruco_detection_status

# See the state machine decision
rostopic echo /state_machine/current_state

# Monitor the sliding window (if you add debug topic)
rostopic echo /aruco/detection_rate
```

## Implementation Notes

- The window is a FIFO queue (first in, first out)
- Detection rate is calculated only when window is full
- State transitions happen immediately when threshold is crossed
- No additional delay after detection becomes consistent
- Works at any camera frame rate (auto-adjusts to actual rate)