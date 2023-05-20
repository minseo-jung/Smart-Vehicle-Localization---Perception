# GPS quality
uint8 NO_FIX=0
uint8 FIX=1
uint8 DIFF_FIX=2
uint8 PPS_FIX=3
uint8 RTK_INT=4
uint8 RTK_FLOAT=5
uint8 DR=6
uint8 MANUAL=7
uint8 SIMULATION=8
uint8 UNKNOWN=9

std_msgs/Header header

float64 latitude
float64 longitude
float64 altitude
float64 heading
float64 latitude_sigma
float64 longitude_sigma
float64 altitude_sigma
float64 heading_sigma
float64 a_sigma 
float64 b_sigma 
uint8 quality 
uint8 number_of_satellites
float32 HDOP 