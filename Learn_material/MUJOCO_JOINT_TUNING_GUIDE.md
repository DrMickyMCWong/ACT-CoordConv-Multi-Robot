# MuJoCo Joint Tuning Guide for Robotic Arms

## Overview
This guide documents the systematic approach to tuning MuJoCo physics parameters for smooth, realistic robotic arm behavior. The goal is to eliminate high-frequency oscillations while maintaining responsive control.

## Problem Analysis

### Common Issues in Untunded Robots
- **High-frequency oscillations** - joints jitter rapidly
- **Overshooting** - joints swing past target positions
- **Instability** - continuous small corrections cause noise
- **Unrealistic stiffness** - robot too rigid, objects fly off during transport
- **Poor tracking** - joints lag behind commanded positions

### Root Causes
1. **Excessive controller gains (kp)** → Aggressive corrections → Oscillations
2. **Missing damping** → No velocity penalty → Sustained oscillations  
3. **Improper gain scaling** → Heavy joints with high gains → Instability
4. **Stiff contact parameters** → Numerical instability

## Tuning Methodology

### Step 1: Identify Problematic Joints
**Method**: Record and plot joint positions (qpos) during operation
- Look for: High-frequency noise, large oscillations, continuous jitter
- **Most common problematic joints**: Base (Joint 1), Wrist joints (Joint 4, 5)
- **Symptoms by joint type**:
  - Base joint → Large amplitude oscillations affecting entire arm
  - Middle joints → Tracking errors, lag behind commands
  - Wrist joints → High-frequency jitter, noise

### Step 2: Mathematical Tuning Approach

#### Controller Gain (kp) Calculation
```
Target Natural Frequency: ωn = √(kp/I)
Desired ωn for robotics: 60-180 rad/s (10-30 Hz)
Therefore: kp = ωn² × I

Where I = joint inertia from XML inertial data
```

#### Damping Calculation  
```
Target Damping Ratio: ζ = 0.6-0.9 (slightly underdamped to critically damped)
damping = 2 × ζ × √(kp × I)
```

#### Joint-Specific Guidelines
| Joint Type | Recommended ωn | Damping Ratio | Notes |
|------------|---------------|---------------|--------|
| Base (Joint 1) | 50-100 rad/s | 0.6-0.8 | Heavy, affects entire arm |
| Shoulder (Joint 2) | 60-120 rad/s | 0.6-0.7 | High load, needs compliance |
| Elbow (Joint 3) | 80-150 rad/s | 0.7-0.8 | Medium load |
| Wrist (Joint 4,5) | 100-200 rad/s | 0.7-0.9 | Light, prone to noise |
| End-effector (Joint 6) | 60-120 rad/s | 0.6-0.8 | Precision required |

### Step 3: Implementation in MuJoCo

#### XML Structure
```xml
<!-- In main XML file (actuator section) -->
<position joint="joint_name" kp="calculated_value" />

<!-- In joint definition XML -->
<joint name="joint_name" damping="calculated_value" />
```

## Case Study: Alicia Robot Tuning

### Original Problems
- Joint 1: Massive oscillations (kp=800 too high for base)
- Joint 2: Very stiff (kp=1600 excessive for shoulder) 
- Joint 4: Poor tracking (kp=10 too low)
- Joint 5: High-frequency noise (kp=50 too high for light wrist)

### Applied Solutions

| Joint | Original kp | Tuned kp | Added Damping | Reasoning |
|-------|-------------|----------|---------------|-----------|
| Joint 1 (Base) | 800 | 150 | 20 | Base joint, high inertia, affects whole arm |
| Joint 2 (Shoulder) | 1600 | 400 | 18 | Heavy load, needed compliance for transport |
| Joint 3 (Elbow) | 800 | 200 | 15 | Medium load, smooth transitions needed |
| Joint 4 (Wrist) | 10 | 80 | 15 | Too low caused tracking errors |
| Joint 5 (Wrist) | 50 | 25 | 8 | Light joint, high-freq noise elimination |
| Joint 6 (End) | 20 | 20 | 0 | Already reasonable, minimal change |

### Mathematical Validation
Example for Joint 1:
```
Original: ωn = √(800/0.000340) = 1536 rad/s (TOO HIGH - 244 Hz)
Tuned: ωn = √(150/0.000340) = 210 rad/s (GOOD - 33 Hz)
Damping ratio: ζ = 20/(2√(150×0.000340)) = 0.65 (WELL DAMPED)
```

## Results and Validation

### Success Metrics
1. **Visual**: Smooth joint trajectories in plots
2. **Functional**: Task completion improves
3. **Realistic**: Action commands differ from actual qpos (filtering effect)
4. **Stable**: No oscillations during static phases

### Expected Behavior After Tuning
- **Policy commands**: May contain noise, high-frequency components
- **Actual joint positions**: Smooth, filtered, physically realistic
- **This difference is DESIRABLE** - physics acts as natural filter

## Key Insights

### Why Action ≠ QPos is Good
- **Real robots filter commands**: Hardware compliance removes impossible motions
- **Physics prevents damage**: Damping protects from jerky movements  
- **Improves performance**: Smooth motion better for manipulation tasks
- **Matches reality**: Real robot arms exhibit this same filtering

### Tuning Philosophy
1. **Start with most problematic joints** (usually base and wrist)
2. **Reduce excessive gains first** (most common issue)
3. **Add appropriate damping** (critical for stability)
4. **Test incrementally** (one joint at a time)
5. **Validate with real task performance** (not just smooth plots)

## Troubleshooting Guide

### If joints are still oscillating:
- Reduce kp further (try 50-70% of current value)
- Increase damping (try 1.5-2x current value)
- Check if joint has adequate frictionloss

### If joints are too sluggish:
- Increase kp moderately (try 1.2-1.5x current value)  
- Reduce damping slightly
- Verify target ωn is in 60-180 rad/s range

### If task performance degrades:
- Check if arm became too compliant during transport phase
- May need to increase kp for load-bearing joints (shoulders)
- Consider different gains for different motion phases

## Best Practices

1. **Always plot joint trajectories** before and after tuning
2. **Tune one joint at a time** to isolate effects
3. **Use mathematical guidelines** as starting points, fine-tune empirically
4. **Test with actual manipulation tasks** not just smooth motion
5. **Document changes and reasons** for future reference
6. **Keep backup of working configurations**

## Advanced Techniques

### Phase 2: Contact Parameter Tuning (if needed)
```xml
<!-- Soften contact if objects fly off -->
<geom solimp="1 0.95 0.001" solref="0.02 1" />
```

### Phase 3: Global Physics Tuning (rarely needed)
```xml
<!-- In main XML if solver issues persist -->
<option timestep="0.002" iterations="100" />
```

## Conclusion

Proper joint tuning transforms a jittery, unrealistic simulation into a smooth, compliant robotic system that matches real hardware behavior. The key insight is that **filtered, smooth joint motion is the goal**, not perfect command tracking.

**Remember**: In real robotics, the best controllers are those that produce smooth, compliant motion - not those that blindly follow every command exactly.