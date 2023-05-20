# mission_state
uint8 STATE_COMEBACK=0
uint8 STATE_CALL_WAIT=1
uint8 STATE_MOVE2START=2
uint8 STATE_START_WAIT=3
uint8 STATE_MOVE2DESTINATION=4
uint8 STATE_TERMINATE_WAIT=5

# action
uint8 ACTION_NONE=0
uint8 ACTION_MOVE=1
uint8 ACTION_STOP=2


# Messages
uint8 action
uint8 state

WGS84       goal_point

float64     goal_point_x
float64     goal_point_y
float64     goal_heading