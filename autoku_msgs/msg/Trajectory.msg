uint8 LANE_KEEPING=0
uint8 CHANGE_LEFT=1
uint8 CHANGE_RIGHT=2
uint8 STOP=3

Header header

bool valid
bool desired
bool on_ego
uint8 behavior
autoku_msgs/SpatialTrajectory spatial_trajectory
autoku_msgs/TemporalTrajectory temporal_trajectory
autoku_msgs/TrajectoryBoundary left_boundary
autoku_msgs/TrajectoryBoundary right_boundary
autoku_msgs/TrajectoryPoint[] point
