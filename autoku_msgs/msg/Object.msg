# Object classification
uint8 CLASSIFICATION_UNKNOWN=0
uint8 CLASSIFICATION_CAR=1
uint8 CLASSIFICATION_BIKE=2
uint8 CLASSIFICATION_MOTORCYCLE=3
uint8 CLASSIFICATION_TRUCK=4
uint8 CLASSIFICATION_PEDESTRIAN=5
uint8 CLASSIFICATION_OTHER_VEHICLE=6

# Object information
uint16 id
uint8 classification
uint8 lane_id

float32 center_x
float32 center_y

float32 center_s
float32 center_d

autoku_msgs/ObjectBoundary[] boundary_points
