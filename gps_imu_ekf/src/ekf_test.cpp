#include "ekf_test.hpp"

ExtendedKalmanFilter::ExtendedKalmanFilter(){
    m_gps_sub = nh.subscribe<sensor_msgs::NavSatFix>("/ublox_gps/fix", 100, &ExtendedKalmanFilter::gpsCallback, this);
    m_imu_sub = nh.subscribe<sensor_msgs::Imu>("/imu/data",1,&ExtendedKalmanFilter::imuCallback,this);
    m_vehicle_speed_sub = nh.subscribe<std_msgs::Float32>("/echo_hello",1,&ExtendedKalmanFilter::speedCallback, this);
    
    gps_path_pub = nh.advertise<nav_msgs::Path>("gps_path", 1);
    ekf_path_pub = nh.advertise<nav_msgs::Path>("ekf_path", 1);
    m_visual_pub = nh.advertise<geometry_msgs::PoseStamped>("heading",1);
    box_pub = nh.advertise<jsk_recognition_msgs::BoundingBox>("car_model",1);

    m_pose_pub = nh.advertise<autoku_msgs::Gnss>("kalman_pose",100);

    gps_input = false, state_init_check = false;
}

ExtendedKalmanFilter::~ExtendedKalmanFilter(){}

void ExtendedKalmanFilter::init(){
    gps_utm.x = 0;
    gps_utm.y = 0;

    vehicle_utm.x = 0;
    vehicle_utm.y = 0;
    vehicle_utm.velocity = 0.5;

    prev_yaw = 0;

    dt = 1.0 / 80;

    prediction_count = 0;
}

void ExtendedKalmanFilter::state_init(){
    if(gps_input && !state_init_check){
        state_init_check = true;
        measure_check = false;
        prev_yaw = atan2((gps_utm.y-gps_utm.prev_y),(gps_utm.x-gps_utm.prev_x));

        x_post(0) = gps_utm.x;
        x_post(1) = gps_utm.y;
        x_post(2) = prev_yaw;
        P_post << 1000, 0, 0,
                    0, 1000, 0,
                    0, 0, 1000;
    }
}

// double ExtendedKalmanFilter::getVehicleSpeed(utm utm){
//     return std::sqrt(std::pow(utm.x - utm.prev_x,2)+std::pow(utm.y - utm.prev_y,2))/gps_dt;
// }

void ExtendedKalmanFilter::gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
    measure_check = true;

    current_pos.lat = msg -> latitude;
    current_pos.lon = msg -> longitude;
    if(!gps_input){
        gps_utm.prev_x = projection.forward(current_pos).x();
        gps_utm.prev_y = projection.forward(current_pos).y();
        gps_input = true;
    }
    else{
        gps_utm.prev_x = gps_utm.x;
        gps_utm.prev_y = gps_utm.y;
    }

    gps_utm.x = projection.forward(current_pos).x();
    gps_utm.y = projection.forward(current_pos).y();

    if(!state_init_check){
        k_pose.latitude = msg->latitude;
        k_pose.longitude = msg->longitude;
    }
}

void ExtendedKalmanFilter::imuCallback(const sensor_msgs::Imu::ConstPtr& msg){
    yaw_rate = msg -> angular_velocity.z;
}

void ExtendedKalmanFilter::speedCallback(const std_msgs::Float32::ConstPtr& msg){
    vehicle_utm.velocity = msg->data;
}

VectorXd ExtendedKalmanFilter::f_k(VectorXd x_post){
    VectorXd fk(N); 
    fk << (x_post(0) + vehicle_utm.velocity * dt * cos(x_post(2))),
        (x_post(1) + vehicle_utm.velocity * dt * sin(x_post(2))),
        (x_post(2) + yaw_rate * dt);
    return fk;
}

void ExtendedKalmanFilter::EKF(){
    current_time = ros::Time::now();
    dt = (current_time - previous_time).toSec();
    
    if(state_init_check){
        Q << 0.1, 0.0, 0.0,          //값 높이면 측정값 비중 증가
             0.0, 0.1, 0.0,
             0.0, 0.0, 0.1;

        R << 1.5, 0.0,               //값 높이면 센서값 비중 증가
             0.0, 1.5;

        I.setIdentity();

        H_jacob << 1.0, 0.0, 0.0,
             0.0, 1.0, 0.0;

        z << gps_utm.x,
            gps_utm.y;

        F_jacob << 1.0, 0.0, -1*vehicle_utm.velocity*dt*sin(vehicle_utm.yaw), 
             0.0, 1.0, vehicle_utm.velocity*dt*cos(vehicle_utm.yaw),
             0.0, 0.0, 1.0;

        x_prior = f_k(x_post);
        f_dr = f_k(x_post);

        P_prior = F_jacob * P_post * F_jacob.transpose() + Q;

        x_post = x_prior;

        P_post = P_prior;

        prediction_count++;

        if(measure_check){
            h << x_prior(0),
                x_prior(1);

            K = P_prior * H_jacob.transpose()*(H_jacob*P_prior*H_jacob.transpose()+R).inverse();
            
            x_post = x_prior + K*(z-h);

            P_post = (I - K * H_jacob) * P_prior;

            prediction_count = 0;

            measure_check = false;
        }

        vehicle_utm.x = x_post(0);
        vehicle_utm.y = x_post(1);
        vehicle_utm.yaw = x_post(2);
    }
    previous_time = current_time;
}

void ExtendedKalmanFilter::publishPose(){
    lanelet::BasicPoint3d utm_point(vehicle_utm.x, vehicle_utm.y, 0);
    lanelet::GPSPoint gps_point = projection.reverse(utm_point);


    k_pose.latitude = gps_point.lat;
    k_pose.longitude = gps_point.lon;

    k_pose.heading = vehicle_utm.yaw;

    m_pose_pub.publish(k_pose);
}

void ExtendedKalmanFilter::Visualization(geometry_msgs::PoseStamped gps_pose, geometry_msgs::PoseStamped ekf_pose){

    gps_pose.header.frame_id = "map";
    gps_pose.header.stamp = ros::Time::now();
    gps_pose.pose.position.x = gps_utm.x;
    gps_pose.pose.position.y = gps_utm.y;
    gps_pose.pose.position.z = 0;
    gps_pose.pose.orientation.x = 0.0;
    gps_pose.pose.orientation.y = 0.0;
    gps_pose.pose.orientation.z = 0.0;
    gps_pose.pose.orientation.w = 1.0;
    gps_path.poses.push_back(gps_pose);
    gps_path.header.stamp = ros::Time::now();
    gps_path.header.frame_id = "map";
    gps_path_pub.publish(gps_path);

    ekf_pose.header.frame_id = "map";
    ekf_pose.header.stamp = ros::Time::now();
    ekf_pose.pose.position.x = vehicle_utm.x;
    ekf_pose.pose.position.y = vehicle_utm.y;
    ekf_pose.pose.position.z = 0;
    ekf_pose.pose.orientation.x = 0.0;
    ekf_pose.pose.orientation.y = 0.0;
    ekf_pose.pose.orientation.z = 0.0;
    ekf_pose.pose.orientation.w = 1.0;
    ekf_path.poses.push_back(ekf_pose);
    ekf_path.header.stamp = ros::Time::now();
    ekf_path.header.frame_id = "map";
    ekf_path_pub.publish(ekf_path);
}

void ExtendedKalmanFilter::visualizeHeading(geometry_msgs::PoseStamped ekf_pose, jsk_recognition_msgs::BoundingBox car_model){

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(vehicle_utm.x, vehicle_utm.y, 0.0));
    tf::Quaternion q;
    q.setRPY(0, 0, vehicle_utm.yaw);
    transform.setRotation(q);
    tfcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "my_car"));

    ekf_pose.header.frame_id = "map";
    ekf_pose.header.stamp = ros::Time::now();
    ekf_pose.pose.position.x = vehicle_utm.x;
    ekf_pose.pose.position.y = vehicle_utm.y;
    ekf_pose.pose.position.z = 0;
    ekf_pose.pose.orientation = tf::createQuaternionMsgFromYaw(vehicle_utm.yaw);

    car_model.header.frame_id = "map";
    car_model.header.stamp = ros::Time::now();
    car_model.dimensions.x = 3.0;
    car_model.dimensions.y = 1.5;
    car_model.dimensions.z = 1.0;
    car_model.pose.position.x = vehicle_utm.x;
    car_model.pose.position.y = vehicle_utm.y;
    car_model.pose.orientation = tf::createQuaternionMsgFromYaw(vehicle_utm.yaw);

    m_visual_pub.publish(ekf_pose);
    box_pub.publish(car_model);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "kalman_test");

    ExtendedKalmanFilter ExtendedKalmanFilter;

    ExtendedKalmanFilter.init();

    ros::Rate loop_rate(80);

    while (ros::ok()){

        ExtendedKalmanFilter.state_init();
        ExtendedKalmanFilter.EKF();
        ExtendedKalmanFilter.Visualization(ExtendedKalmanFilter.gps_pose, ExtendedKalmanFilter.ekf_pose);
        ExtendedKalmanFilter.visualizeHeading(ExtendedKalmanFilter.ekf_pose, ExtendedKalmanFilter.car_model);
        ExtendedKalmanFilter.publishPose();

        ros::spinOnce();
        loop_rate.sleep();

    }

    return 0;
}