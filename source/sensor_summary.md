# UAV Navigation System: Sensor Analysis & Measurement Units

## Overview of Sensor Data

Based on your project code and documentation, the following sensors and their respective measurement units are used in your UAV navigation system:

## 1. IMU Data (`RawAccel.csv`)

```
Sensor: Accelerometer
File: RawAccel.csv
```

| Column | Unit | Description |
|--------|------|-------------|
| `Timpstemp` | μs | Timestamp in microseconds |
| `x` | m/s² | X-axis linear acceleration |
| `y` | m/s² | Y-axis linear acceleration |
| `z` | m/s² | Z-axis linear acceleration |
| `temperature` | °C | Sensor temperature |

## 2. Gyroscope Data (`RawGyro.csv`)

```
Sensor: Gyroscope
File: RawGyro.csv
```

| Column | Unit | Description |
|--------|------|-------------|
| `Timpstemp` | μs | Timestamp in microseconds |
| `x` | rad/s | X-axis angular velocity |
| `y` | rad/s | Y-axis angular velocity |
| `z` | rad/s | Z-axis angular velocity |
| `temperature` | °C | Sensor temperature |

## 3. GPS Data (`OnboardGPS.csv`)

```
Sensor: GPS
File: OnboardGPS.csv
```

| Column | Unit | Description |
|--------|------|-------------|
| `Timpstemp` | μs | Timestamp in microseconds |
| `vel_n_m_s` | m/s | North velocity component |
| `vel_e_m_s` | m/s | East velocity component |
| `vel_d_m_s` | m/s | Down velocity component |

## 4. Ground Truth Data (`OnboardPose.csv`)

```
Sensor: Combined system (reference)
File: OnboardPose.csv
```

| Column | Unit | Description |
|--------|------|-------------|
| `Timpstemp` | μs | Timestamp in microseconds |
| `Omega_x` | rad/s | X-axis angular velocity |
| `Omega_y` | rad/s | Y-axis angular velocity |
| `Omega_z` | rad/s | Z-axis angular velocity |
| `Accel_x` | m/s² | X-axis linear acceleration |
| `Accel_y` | m/s² | Y-axis linear acceleration |
| `Accel_z` | m/s² | Z-axis linear acceleration |
| `Vel_x` | m/s | X-axis velocity |
| `Vel_y` | m/s | Y-axis velocity |
| `Vel_z` | m/s | Z-axis velocity |
| `AccBias_x` | m/s² | X-axis accelerometer bias |
| `AccBias_y` | m/s² | Y-axis accelerometer bias |
| `AccBias_z` | m/s² | Z-axis accelerometer bias |
| `Attitude_w` | unitless | Quaternion w component |
| `Attitude_x` | unitless | Quaternion x component |
| `Attitude_y` | unitless | Quaternion y component |
| `Attitude_z` | unitless | Quaternion z component |
| `Height` | m | Height above reference |

## 5. Error State Vector (Kalman Filter)

From your document `kalman-documentation.md`, the error state vector includes:

| Component | Unit | Description |
|-----------|------|-------------|
| φE, φN, φU | rad | Attitude errors (roll, pitch, yaw) |
| δυE, δυN, δυU | m/s | Velocity errors (East, North, Up) |
| δL, δλ, δh | m | Position errors (latitude, longitude, height) |
| εx, εy, εz | rad/s | Gyroscope bias errors |
| ∇x, ∇y, ∇z | m/s² | Accelerometer bias errors |

## Performance Metrics

Based on your thesis proposal (Section 4.1), the following metrics are defined:

1. **Integrated Settling Error (ISE)**
   - Unit: m·s (meter-seconds)
   - Formula: `IRE = ∫|desired(t) - actual(t)| dt`
   - Purpose: Measures cumulative position error over recovery time

2. **Mean Absolute Difference Kalman Gain (MAD_K)**
   - Unit: unitless
   - Formula: `MAD_K = (1/N)∑|K_standard,i - K_assisted,i|`
   - Purpose: Compares standard vs. LNN-assisted Kalman gain

3. **Root Mean Square Error (RMSE)**
   - Unit: varies by component (m for position, rad for attitude)
   - Formula: `RMSE = √((1/N)∑(ŷ_i - y_i)²)`
   - Purpose: Measures prediction accuracy

4. **2D and Z-Direction Errors**
   - Units: m (meters)
   - Formulas:
     - `E_2D = √((x_desired - x_actual)² + (y_desired - y_actual)²)`
     - `E_Z = |z_desired - z_actual|`
   - Purpose: Separate horizontal and vertical position accuracy assessment

## Sampling Rates

According to your code and various documents:

- IMU data: ~10 Hz (9.97 Hz)
- Gyroscope data: ~30 Hz (29.91 Hz)
- GPS data: Variable, resampled to 10 Hz
- Ground truth data: ~50 Hz (49.78 Hz)
- Aligned data: 10 Hz (100ms intervals)

## References

1. Chen, K. et al. (2024). "Research on Kalman Filter Fusion Navigation Algorithm Assisted by CNN-LSTM Neural Network." Applied Sciences, 14, 5493.
2. Your thesis proposal document, Section 4.1 (pages 5-6)
3. `kalman-documentation.md` from your project files
4. `load_data` function from your code