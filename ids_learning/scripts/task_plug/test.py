#!usr/bin/env python
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
import os
from envs.mrobot import MRobot
from agents.dqn import DQN
from agents.ppo import PPO
from envs.env_socket_plug import SocketPlugEnv
import argparse
import matplotlib.pyplot as plt
import csv
import pandas as pd

# np.random.seed(111)

"""
Test full plugging process
"""
class SocketPlugFullTest:
    def __init__(self, model):
        self.model = model
        self.camera = RSD435('camera')
        self.ftSensor = FTSensor('ft_endeffector')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self.socketDetector = ObjectDetector(topic='detection',type=4)
        self.outletDetector = ObjectDetector(topic='detection',type=3)

    def robot_pose(self):
        pos = self.poseSensor.robot()
        noise = np.random.uniform(size=3)
        x = pos[0] + 0.1*(noise[0]-0.5) # position noise = 0.1 m
        y = pos[1] + 0.1*(noise[1]-0.5) # position noise = 0.1 m
        yaw = pos[2][2] + 0.1*(noise[2]-0.5) # orientation noise = 0.1 rad
        return x,y,yaw

    def reset(self,rx,ry,rt):
        print("reset robot",rx,ry,rt)
        self.driver.stop()
        self.robotPoseReset.reset_robot(rx,ry,rt)
        self.fdController.set_position(hk=1.57,vs=0,hs=0,pg=0.03)
        self.fdController.lock_hook()
        self.fdController.lock_plug()
        self.socketDetector.reset()
        self.outletDetector.reset()

    def terminate(self):
        print("terminate test.")
        self.fdController.unlock_hook()
        self.fdController.unlock_plug()

    def search_outlet(self,vz):
        print("searching outlet")
        self.driver.stop()
        rospy.sleep(1)
        rate = rospy.Rate(10)
        while not self.outletDetector.ready():
            self.driver.drive(0.0,vz)
            rate.sleep()
        # put the target in center of view
        last = self.outletDetector.get_detect_info()[-1]
        cx = 0.5*(last.l+last.r)
        while abs(cx-320) > 10:
            self.driver.drive(0.0, np.sign(320-cx)*vz)
            last = self.outletDetector.get_detect_info()[-1]
            cx = 0.5*(last.l+last.r)
        self.driver.stop()
        # calculate target position
        dist2outlet = last.z
        rx, ry, rt = self.robot_pose()
        ox = rx + dist2outlet*np.cos(rt)
        oy = ry + dist2outlet*np.sin(rt)
        print("robot at ({:.3f},{:.3f}) with yaw {:.3f}, detected outlet at ({:.3f},{:.3f}) with normal {:.3f}".format(rx,ry,rt,ox,oy,last.nx))
        return (ox,oy)

    def align_with_outlet(self,outlet,tolerance=0.01):
        print("aligning with outlet")
        detect = outlet
        while abs(detect.nx) > tolerance:
            self.driver.drive(0.0, -np.sign(detect.nx)*2*np.pi)
            rospy.sleep(5)
            self.driver.drive(2.0, 0.0)
            rospy.sleep(3)
            self.driver.drive(0.0, np.sign(detect.nx)*2*np.pi)
            rospy.sleep(3)
            detect = self.outletDetector.get_detect_info()[-1]
        self.driver.stop()
        return detect

    def align_endeffector(self,detect):
        print("aligning endeffector")
        hpos = self.fdController.hslider_pos()
        self.fdController.move_hslider_to(hpos-detect.x)
        vpos = self.fdController.vslider_pos()
        self.fdController.move_vslider_to(vpos-detect.y+0.072)
        return self.socketDetector.info

    def move_closer(self,info,distance=0.8):
        print("moving closer")
        detect = info
        rate = rospy.Rate(10)
        while detect.z > distance:
            self.driver.drive(0.5,0.0)
            detect = self.socketDetector.info
            rate.sleep()
        self.driver.stop()
        return self.socketDetector.info

    def approach(self,info,distance=0.3):
        print("approaching")
        d0 = info.z
        w0 = info.r-info.l
        h0 = info.b-info.t
        u0 = (info.l+info.r)/2
        v0 = (info.t+info.b)/2
        d = d0
        while d > distance:
            self.driver.drive(0.5,0.0)
            detect = self.socketDetector.info
            w = detect.r-detect.l
            h = detect.b-detect.t
            d = d0*((w0/w)+(h0/h))/2
        self.driver.stop()
        return self.socketDetector.info

    def perform_plug(self):
        print("performing plug")
        done, step = False, 0
        image = self.camera.grey_arr((64,64))
        force = self.ftSensor.forces()
        while not done or step < 60:
            obs = dict(image=image, force=force)
            action = self.agent.policy(obs,0.0)
            act = self.get_action(action)
            print(act)
            hpos = self.fdController.hslider_pos()
            self.fdController.move_hslider_to(hpos+act[0])
            vpos = self.fdController.vslider_pos()
            self.fdController.move_vslider_to(vpos+act[1])
            step += 1
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02
            if not self.success and not self.fail:
                force = self.plug()
            else:
                break
        return self.success

    def get_action(self,action):
        sh,sv=0.001,0.001
        act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
        return act_list[action]

    def plug(self, f_max=20):
        # print("plugging")
        self.fdController.lock_vslider()
        self.fdController.lock_hslider()
        forces, dist1, dist2 = self.ftSensor.forces(), 0, 0
        while forces[0] > -f_max and abs(forces[1]) < 10 and abs(forces[2]+9.8) < 10:
            self.driver.drive(0.2,0.0)
            forces = self.ftSensor.forces()
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02
            if self.success or self.fail:
                break
            rospy.sleep(0.01)
        self.driver.stop()
        self.curr_dist = dist2
        self.fdController.unlock_hslider()
        self.fdController.unlock_vslider()
        return forces

    def dist2goal(self):
        bpPos = self.poseSensor.bumper()
        dist1 = bpPos[1] - self.goal[1]
        dist2 = np.sqrt((bpPos[0]-self.goal[0])**2 + (bpPos[2]-self.goal[2])**2)
        return dist1, dist2

    def run(self,rx,ry,rt):
        self.reset(rx,ry,rt)
        outletPos = self.search_outlet(vz=2*np.pi)
        # outlet = self.align_with_outlet(outletPos,tolerance=0.005)
        # detect = self.align_endeffector(detect)
        # detect = self.move_closer(detect,distance=0.8)
        # detect = self.align_normal(detect,tolerance=0.001)
        # detect = self.align_endeffector(detect)
        # detect = self.approach(detect,distance=0.2)
        # isSuccess = self.perform_plug()
        self.terminate()
        return

"""
RUN full test
"""
def run_full_test(agent, policy, index, rad):
    print("run full test", agent, policy, index)
    if agent != 'dqn':
        print("undefined agent type")
        return

    model_path = os.path.join(sys.path[0],"../policy/socket_plug/")+policy+"/q_net/"+str(index)
    print("load model from", model_path)
    model = DQN((64,64,1),3,2,8,gamma=0.99,lr=2e-4,update_freq=500)
    model.load(model_path)
    test = SocketPlugFullTest(model)
    rx = 2 + 2*(rad[0]-0.5)
    ry = 1 + (rad[1]-0.5)
    rt = 2*np.pi*(rad[2]-0.5)
    test.run(rx,ry,rt)
    print("terminated")


"""
TEST socket plug
"""
class SocketPlugTest:
    def __init__(self,env,agent=None,policy=None,index=None):
        self.env = env
        self.agent = agent
        self.policy = policy
        self.load_model(agent,policy,index)

    def load_model(self,agent,policy,index):
        if agent == 'dqn':
            model_path = os.path.join(sys.path[0],"../policy/socket_plug/")+policy+"/q_net/"+str(index)
            print("load model from", model_path)
            self.model = DQN((64,64,1),3,2,8,gamma=0.99,lr=2e-4,update_freq=500)
            self.model.load(model_path)
        else:
            print("undefined agent")
            self.model = None


    def action(self,obs):
        if self.agent is None:
            return np.random.randint(self.env.action_space.n)
        else:
            return self.model.policy(obs)

    def run(self, init_rad):
        positions = []
        self.env.set_init_random(init_rad)
        obs, info = self.env.reset()
        init = info["plug"]
        positions.append(init)
        done, step = False, 0
        while not done and step < 50:
            act = self.action(obs)
            obs, rew, done, info = self.env.step(act)
            positions.append(info["plug"])
            if step == 0:
                self.env.robot.ftPlug.reset()
            step += 1
        return self.env.success, step, self.env.robot.ftPlug.record, positions, init

"""
RUN a single plug test for a policy on a target
"""
def run_plug_test(env, agent, policy, index, target, init_rads):
    print("run plug test", agent, policy, index, target)
    data_dir = os.path.join(sys.path[0],'data',policy+'_'+str(target))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    try_count = len(init_rads)
    test = SocketPlugTest(env,agent,policy,index)
    success_steps, results = [],[]
    for i in range(try_count):
        success, step, forces, positions, init = test.run(init_rads[i])
        pd.DataFrame(forces).to_csv(data_dir+'/forces_'+str(i)+'.csv', index=False)
        pd.DataFrame(positions).to_csv(data_dir+'/positions_'+str(i)+'.csv', index=False)
        if success:
            success_steps.append(step)
        results.append((success,init[0],init[1],init[2],init[3],step))
        print("plug", i+1, "/", try_count, "steps", step, "success", success, "success_counter", len(success_steps))
    pd.DataFrame(results).to_csv(data_dir+'/test_data.csv', index=False)
    print("sucess rate", len(success_steps)/try_count, "average steps", np.mean(success_steps))
    return len(success_steps), np.mean(success_steps)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='plug') # 'plug','full'
    parser.add_argument('--agent', type=str, default='dqn') # dqn, ppo, none
    parser.add_argument('--policy', type=str, default='binary') # binary, greyscale, blind
    parser.add_argument('--iter', type=int, default=6850) # binary 6850, raw 6700
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('plug_test', anonymous=True)
    if args.type == 'plug':
        target_count,try_count = 8,30
        rads = np.random.uniform(size=(target_count,try_count,4))
        policy_list = []
        if args.policy is None:
            policy_list = ['binary', 'greyscale', 'blind', 'raw']
        else:
            policy_list = [args.policy]

        test_res = []
        env = SocketPlugEnv(continuous=False)
        for policy in policy_list:
            env.set_vision_type(policy)
            for i in range(target_count):
                env.set_goal(2)
                success_count, mean_steps = run_plug_test(env,args.agent,policy,args.iter,i,rads[i])
                test_res.append((policy,i,success_count,mean_steps))
        for res in test_res:
            print("policy", res[0], "target outlet", res[1], "success count", res[2], "average steps", res[3])
    else:
        print("full plugging test")
        rad = np.random.uniform(size=3)
        run_full_test(args.agent,args.policy,args.iter,rad)
