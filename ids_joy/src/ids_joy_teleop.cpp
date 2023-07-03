# include <ros/ros.h>
# include <geometry_msgs/Twist.h>
# include <sensor_msgs/Joy.h>
# include <std_msgs/Float32MultiArray.h>
# include <std_msgs/Int32.h>
# include <std_msgs/Float64.h>
# include <control_msgs/JointControllerState.h>

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
  } axes;

  struct
  {
    Button stop;
    Button up;
    Button down;
    Button left;
    Button right;
  } buttons;


  void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
  void jh_cb(const control_msgs::JointControllerState::ConstPtr& msg);
  void jv_cb(const control_msgs::JointControllerState::ConstPtr& msg);
  void motorCallback(const std_msgs::Float32MultiArray::ConstPtr& status);
  double getAxis(const sensor_msgs::JoyConstPtr& joy, const Axis &axis);
  bool getButton(const sensor_msgs::JoyConstPtr& joy, const Button &button);
  void stop();

  int simulation;
  double jh, jv;
  ros::NodeHandle nh;
  ros::Subscriber joy_sub, hm_sub, vm_sub;
  ros::Publisher base_pub, hm_pub, vm_pub;

};


TeleopIDS::TeleopIDS()
{
  ros::param::get("/teleop/simulation",simulation);
  ros::param::get("/teleop/linear_axis", axes.linear.axis);
  ros::param::get("/teleop/angular_axis", axes.angular.axis);
  ros::param::get("/teleop/stop_btn", buttons.stop.button);
  ros::param::get("/teleop/up_btn", buttons.up.button);
  ros::param::get("/teleop/down_btn", buttons.down.button);
  ros::param::get("/teleop/left_btn", buttons.left.button);
  ros::param::get("/teleop/right_btn", buttons.right.button);
  ros::param::get("/teleop/linear_scale", axes.linear.factor);
  ros::param::get("/teleop/angular_scale", axes.angular.factor);

  joy_sub = nh.subscribe<sensor_msgs::Joy>("joy",3,&TeleopIDS::joyCallback,this);
  base_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel",1);
  if (simulation) {
    hm_pub = nh.advertise<std_msgs::Float64>("joint_hslider_controller/command",1);
    vm_pub = nh.advertise<std_msgs::Float64>("joint_vslider_controller/command",1);
    hm_sub = nh.subscribe<control_msgs::JointControllerState>("joint_hslider_controller/state",3,&TeleopIDS::jh_cb,this);
    vm_sub = nh.subscribe<control_msgs::JointControllerState>("joint_vslider_controller/state",3,&TeleopIDS::jv_cb,this);
  }
  else {
    hm_pub = nh.advertise<std_msgs::Int32>("robo1_cmd",1);
    vm_pub = nh.advertise<std_msgs::Int32>("robo3_cmd",1);
  }
}

void TeleopIDS::jh_cb(const control_msgs::JointControllerState::ConstPtr& msg)
{
  jh = double(msg->process_value);
}

void TeleopIDS::jv_cb(const control_msgs::JointControllerState::ConstPtr& msg)
{
  jv = double(msg->process_value);
}

void TeleopIDS::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
  // validate before start
  if (getButton(joy, buttons.stop))
  {
    stop();
  }
  else if (getButton(joy, buttons.up))
  {
    if (simulation) {
      std_msgs::Float64 msg;
      msg.data = jv+0.001;
      vm_pub.publish(msg);
    } else {
      std_msgs::Int32 msg;
      msg.data = 1;
      vm_pub.publish(msg);
    }
  }
  else if (getButton(joy, buttons.down))
  {
    if (simulation) {
      std_msgs::Float64 msg;
      msg.data = jv-0.001;
      vm_pub.publish(msg);
    } else {
      std_msgs::Int32 msg;
      msg.data = -1;
      vm_pub.publish(msg);
    }
  }
  else if (getButton(joy, buttons.left))
  {
    if (simulation) {
      std_msgs::Float64 msg;
      msg.data = jh+0.001;
      hm_pub.publish(msg);
    } else {
      std_msgs::Int32 msg;
      msg.data = 1;
      hm_pub.publish(msg);
    }
  }
  else if (getButton(joy, buttons.right))
  {
    if (simulation) {
      std_msgs::Float64 msg;
      msg.data = jh-0.001;
      hm_pub.publish(msg);
    } else {
      std_msgs::Int32 msg;
      msg.data = -1;
      hm_pub.publish(msg);
    }
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
  if (simulation) {
    std_msgs::Float64 msg;
    msg.data = jv;
    vm_pub.publish(msg);
    msg.data = jh;
    hm_pub.publish(msg);
  } else {
    std_msgs::Int32 msg;
    msg.data = 0;
    vm_pub.publish(msg);
    hm_pub.publish(msg);
  }
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
