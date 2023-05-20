# direction type
int8 FORWARD=1
int8 BACKWARD=-1
# gear type
uint8 PARKING=0
uint8 REVERSE=1
uint8 NEUTRAL=2
uint8 DRIVE=3
# turn signal type
uint8 NONE=0
uint8 LEFT=1
uint8 RIGHT=2
uint8 HAZARD=3
# driving mode
uint8 COMFORT=0
uint8 SPORT=1
# operation mode
uint8 MANUAL=0
uint8 AUTONOMOUS=1
# autonomous mode
uint8 READY=0
uint8 RUN=1
uint8 PAUSE=2

# std_msgs/Header
Header header

# odometry state
float32 x
float32 y
float32 z
float32 yaw
float32 pitch
float32 roll
float32 yaw_rate
float32 pitch_rate
float32 roll_rate
float32 x_sigma
float32 y_sigma
float32 z_sigma
float32 yaw_sigma
float32 pitch_sigma
float32 roll_sigma
float32 yaw_rate_sigma
float32 pitch_rate_sigma
float32 roll_rate_sigma
float32 speed
float32 acceleration
float32 steering_angle
int32 wheel_pulse_fl
int32 wheel_pulse_fr
int32 wheel_pulse_rl
int32 wheel_pulse_rr
int32 wheel_speed_fl
int32 wheel_speed_fr
int32 wheel_speed_rl
int32 wheel_speed_rr
int32 wheel_direction_fl
int32 wheel_direction_fr
int32 wheel_direction_rl
int32 wheel_direction_rr
int8 direction
uint8 gear
uint8 turn_signal
uint8 door_open
uint8 driving_mode
uint8 operation_mode
uint8 autonomous_mode
