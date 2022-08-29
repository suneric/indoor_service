# include <ros/ros.h>
# include <geometry_msgs/Twist.h>
# include <sensor_msgs/Joy.h>
# include <std_msgs/Float32MultiArray.h>
# include <std_msgs/Int32.h>

// refereces
// -http://wiki.ros.org/ps3joy
// -http://wiki.ros.org/joy/Tutorials

class TeleopIDS
{
public:
  TeleopIDS();

private:

  struct Axis
  {
    Axis():axis(0),factor(0.0),offset(0.0){}
    int axis;
    double factor;
    double offset;
  };

  struct Button
  {
    Button():button(0){}
    int button;
  };

  struct
  {
    Axis linear;
    Axis angular;
    Axis horizontal;
    Axis vertical;
  } axes;

  struct
  {
    Button stop;
    Button forward;
    Button backward;
  } buttons;


  void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
  void motorCallback(const std_msgs::Float32MultiArray::ConstPtr& status);
  double getAxis(const sensor_msgs::JoyConstPtr& joy, const Axis &axis);
  bool getButton(const sensor_msgs::JoyConstPtr& joy, const Button &button);
  void stop();

  ros::NodeHandle nh;
  ros::Subscriber joy_sub, motor_sub;
  ros::Publisher base_pub, robo1_pub, robo2_pub, robo3_pub;
};


TeleopIDS::TeleopIDS()
{
  ros::param::get("/teleop/linear_axis", axes.linear.axis);
  ros::param::get("/teleop/angular_axis", axes.angular.axis);
  ros::param::get("/teleop/vertical_axis", axes.vertical.axis);
  ros::param::get("/teleop/horizontal_axis", axes.horizontal.axis);
  ros::param::get("/teleop/stop_btn", buttons.stop.button);
  ros::param::get("/teleop/forward_btn", buttons.forward.button);
  ros::param::get("/teleop/backward_btn", buttons.backward.button);
  ros::param::get("/teleop/linear_scale", axes.linear.factor);
  ros::param::get("/teleop/angular_scale", axes.angular.factor);
  ros::param::get("/teleop/horizontal_scale", axes.horizontal.factor);
  ros::param::get("/teleop/vertical_scale", axes.vertical.factor);

  joy_sub = nh.subscribe<sensor_msgs::Joy>("joy", 10, &TeleopIDS::joyCallback, this);
  motor_sub = nh.subscribe<std_msgs::Float32MultiArray>("robomotor_status", 10, &TeleopIDS::motorCallback, this);
  base_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel",1);
  robo1_pub = nh.advertise<std_msgs::Int32>("robo1_cmd",1);
  robo2_pub = nh.advertise<std_msgs::Int32>("robo2_cmd",1);
  robo3_pub = nh.advertise<std_msgs::Int32>("robo3_cmd",1);
}

void TeleopIDS::motorCallback(const std_msgs::Float32MultiArray::ConstPtr& status)
{
  // std::cout << "status" << std::endl;
}

void TeleopIDS::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
  // validate before start
  if (getButton(joy, buttons.stop))
  {
    stop();
  }

  // motor driving
  bool push = getButton(joy, buttons.forward);
  bool pull = getButton(joy, buttons.backward);
  if (push || pull)
  {
    std_msgs::Int32 msg;
    msg.data = push ? 1 : -1;
    robo2_pub.publish(msg);
  }

  double h = getAxis(joy, axes.horizontal);
  if (std::abs(h) > 0)
  {
    std_msgs::Int32 msg;
    msg.data = (h > 0) ? 1 : -1;
    robo1_pub.publish(msg);
  }

  double v = getAxis(joy, axes.vertical);
  if (std::abs(v) > 0)
  {
    std_msgs::Int32 msg;
    msg.data = (v > 0) ? 1 : -1;
    robo3_pub.publish(msg);
  }

  // base driving
  double l = getAxis(joy, axes.linear);
  double a = getAxis(joy, axes.angular);
  if (std::abs(l) > 0 || std::abs(a) > 0)
  {
    geometry_msgs::Twist twist;
    twist.angular.z = a;
    twist.linear.x = l;
    base_pub.publish(twist);
  }
}

double TeleopIDS::getAxis(const sensor_msgs::JoyConstPtr &joy, const Axis &axis)
{
  if (axis.axis < 0 || std::abs(axis.axis) > joy->axes.size())
  {
    ROS_ERROR_STREAM("Axis " << axis.axis << " out of range, joy has " << joy->axes.size() << " axes");
    return 0;
  }

  double output = joy->axes[std::abs(axis.axis)] * axis.factor + axis.offset;
  return output;
}

bool TeleopIDS::getButton(const sensor_msgs::JoyConstPtr &joy, const Button &button)
{
  if (button.button < 0 || button.button > joy->buttons.size())
  {
    ROS_ERROR_STREAM("Button " << button.button << " out of range, joy has " << joy->buttons.size() << " buttons");
    return false;
  }

  return joy->buttons[button.button];
}

void TeleopIDS::stop()
{
  std_msgs::Int32 stop;
  stop.data = 0;
  robo1_pub.publish(stop);
  robo2_pub.publish(stop);
  robo3_pub.publish(stop);

  geometry_msgs::Twist twist;
  twist.angular.z = 0;
  twist.linear.x = 0;
  base_pub.publish(twist);
}

//////////////////////////////////////////////
int main(int argc, char** argv)
{
  ros::init(argc, argv, "teleop_ids");
  TeleopIDS teleop_ids;
  ros::spin();
  return 0;
}
