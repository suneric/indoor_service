import os
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import argparse
from robot.mrobot import MRobot
from robot.jrobot import JazzyRobot

def collect(robot, simulation, t, dump_dir):
    idx = 0
    while not rospy.is_shutdown():
        frc = robot.hook_forces()
        img = robot.camARD2.color_image() if simulation else robot.camARD1.color_image()
        img_file = os.path.join(dump_dir,"_{}_{:.4f}_{:.4f}_{:.4f}".format(idx,frc[0],frc[1],frc[2]))
        cv.imwrite(dump_dir,img)
        rospy.sleep(t)
        idx += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('door_pull_test', anonymous=True)
    isSim = args.simulation == 1
    robot = MRobot() if isSim else JazzyRobot()
    dump_dir = os.path.join(sys.path[0],"../../dump/collection/")
    collect(robot,isSim,0.5,dump_dir)
