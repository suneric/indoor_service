import numpy as np
import rospy
import gym # https://github.com/openai/gym/blob/master/gym/core.py
# from gym.utils import seeding
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class GazeboConnection():

    def __init__(self, start_init_physics_parameters, reset_world_or_sim):

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Setup the Gravity Controle system
        service_name = '/gazebo/set_physics_properties'
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))

        self.set_physics = rospy.ServiceProxy(service_name, SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()
        # We always pause the simulation, important for legged robots learning
        self.pauseSim()

    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")


    def resetSim(self):
        """
        This was implemented because some simulations, when reseted the simulation
        the systems that work with TF break, and because sometime we wont be able to change them
        we need to reset world that ONLY resets the object position, not the entire simulation
        systems.
        """
        if self.reset_world_or_sim == "SIMULATION":
            rospy.logwarn("SIMULATION RESET")
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            rospy.logwarn("WORLD RESET")
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logwarn("NO RESET SIMULATION SELECTED")
        else:
            rospy.logwarn("WRONG Reset Option:"+str(self.reset_world_or_sim))

    def resetSimulation(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_world service call failed")

    def init_values(self):

        self.resetSim()

        if self.start_init_physics_parameters:
            rospy.logwarn("Initialising Simulation Physics Parameters")
            self.init_physics_parameters()
        else:
            rospy.logerr("NOT Initialising Simulation Physics Parameters")

    def init_physics_parameters(self):
        """
        We initialise the physics parameters of the simulation, like gravity,
        friction coeficients and so on.
        """
        self._time_step = Float64(0.001)
        self._max_update_rate = Float64(1000.0)

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics()
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 50
        self._ode_config.sor_pgs_w = 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()


    def update_gravity_call(self):

        self.pauseSim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rospy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rospy.logdebug("Gravity Update Result==" + str(result.success) + ",message==" + str(result.status_message))

        self.unpauseSim()

    def change_gravity(self, x, y, z):
        self._gravity.x = x
        self._gravity.y = y
        self._gravity.z = z

        self.update_gravity_call()

class GymGazeboEnv(gym.Env):
  """
  Gazebo env converts standard openai gym methods into Gazebo commands

  To check any topic we need to have the simulations running, we need to do two things:
    1)Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
      that are pause for whatever the reason
    2)If the simulation was running already for some reason, we need to reset the controlers.
      This has to do with the fact that some plugins with tf, dont understand the reset of the simulation and need to be reseted to work properly.
  """
  def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="WORLD"):
    # To reset Simulations
    rospy.logdebug("START init RobotGazeboEnv")
    self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim)
    # self.seed()
    rospy.logdebug("END init RobotGazeboEnv")

  # Env methods
  # def seed(self, seed=None):
  #   self.np_random, seed = seeding.np_random(seed)
  #   return [seed]

  def step(self, action):
    """
    Gives env an action to enter the next state,
    obs, reward, done, info = env.step(action)
    """
    # Convert the action num to movement action
    self.gazebo.unpauseSim()
    self._take_action(action)
    self.gazebo.pauseSim()
    obs = self._get_observation()
    reward = self._compute_reward()
    done = self._is_done()
    info = self._post_information()

    return obs, reward, done, info

  def reset(self):
    """
    obs, info = env.reset()
    """
    # self.gazebo.pauseSim()
    rospy.logdebug("Reseting RobotGazeboEnvironment")
    self._reset_sim()
    obs = self._get_observation()
    rospy.logdebug("END Reseting RobotGazeboEnvironment")
    return obs

  def close(self):
    """
    Function executed when closing the environment.
    Use it for closing GUIS and other systems that need closing.
    :return:
    """
    rospy.logwarn("Closing RobotGazeboEnvironment")
    rospy.signal_shutdown("Closing RobotGazeboEnvironment")

  def _reset_sim(self):
    """Resets a simulation
    """
    rospy.logdebug("START robot gazebo _reset_sim")
    self.gazebo.unpauseSim()
    self.gazebo.resetSim()
    self._set_init()
    self.gazebo.pauseSim()
    self.gazebo.unpauseSim()
    self._check_all_systems_ready()
    self.gazebo.pauseSim()
    rospy.logdebug("END robot gazebo _reset_sim")

    return True

  def _set_init(self):
    """Sets the Robot in its init pose
    """
    raise NotImplementedError()

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    raise NotImplementedError()

  def _get_observation(self):
    """Returns the observation.
    """
    raise NotImplementedError()

  def _post_information(self):
    """Returns the info.
    """
    raise NotImplementedError()

  def _take_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _is_done(self):
    """Indicates whether or not the episode is done ( the robot has fallen for example).
    """
    raise NotImplementedError()

  def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    raise NotImplementedError()

  def _env_setup(self, initial_qpos):
    """Initial configuration of the environment. Can be used to configure initial state
    and extract information from the simulation.
    """
    raise NotImplementedError()
