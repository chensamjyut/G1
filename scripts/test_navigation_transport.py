"""Actual C++ ROS planner, SYNTHETIC odometry/cloud. Never a robot success test.
Run under ROS Humble with ROS_LOCALHOST_ONLY=1 and an isolated ROS_DOMAIN_ID.
"""
import argparse,json,os,subprocess,time,signal
from pathlib import Path
import rclpy
from nav_msgs.msg import Odometry,Path as NavPath
from geometry_msgs.msg import PointStamped,PoseStamped,TwistStamped
from std_msgs.msg import Bool,String,Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True);a=p.parse_args();root=a.root.resolve()
    if os.getenv('ROS_LOCALHOST_ONLY')!='1' or os.getenv('ROS_DOMAIN_ID')!='177':raise RuntimeError('requires_isolated_test_domain_177_loopback')
    install=root/'sim_runtime/navigation/install/local_planner';exe=install/'lib/local_planner';share=install/'share/local_planner';children=[]
    rclpy.init();node=rclpy.create_node('release_synthetic_transport_test');seen={'path_count':0,'nonempty_path_count':0,'commands':0,'status_count':0}
    def paths(m):seen['path_count']+=1;seen['nonempty_path_count']+=int(len(m.poses)>1)
    def commands(m):seen['commands']+=1
    def status(m):seen['status_count']+=1
    subs=[node.create_subscription(NavPath,'/path',paths,10),node.create_subscription(TwistStamped,'/cmd_vel',commands,10),node.create_subscription(String,'/fastlio_far/goal_status',status,10)]
    pubs={k:node.create_publisher(t,k,10) for k,t in [('/state_estimation',Odometry),('/registered_scan',PointCloud2),('/way_point',PointStamped),('/goal_pose',PoseStamped),('/localization_ready',Bool)]}
    try:
        common=['--ros-args','--params-file',str(share/'config/omniDir.yaml'),'--params-file',str(share/'config/unitree/unitree_g1.yaml')]
        children.append(subprocess.Popen([str(exe/'localPlanner'),*common,'-p','pathFolder:='+str(share/'paths'),'-p','useTerrainAnalysis:=false','-p','autonomyMode:=true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        children.append(subprocess.Popen([str(exe/'inspectionPathFollower'),*common,'-p','realRobot:=false','-p','useFinalGoalPose:=true','-p','autonomyMode:=true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        start=time.monotonic()
        while time.monotonic()-start<15:
            if any(c.poll() is not None for c in children):raise RuntimeError('ROS_node_exited')
            stamp=node.get_clock().now().to_msg();o=Odometry();o.header.stamp=stamp;o.header.frame_id='map';o.pose.pose.orientation.w=1.;pubs['/state_estimation'].publish(o)
            g=PointStamped();g.header=o.header;g.point.x=2.;pubs['/way_point'].publish(g)
            pose=PoseStamped();pose.header=o.header;pose.pose.position.x=2.;pose.pose.orientation.w=1.;pubs['/goal_pose'].publish(pose)
            pubs['/localization_ready'].publish(Bool(data=True))
            # Distant sparse obstacle points; zero actual robot or sensor involved.
            cloud=point_cloud2.create_cloud_xyz32(o.header,[(5.,-2.+i*.1,0.5) for i in range(40)])
            pubs['/registered_scan'].publish(cloud);rclpy.spin_once(node,timeout_sec=.05)
        result={'kind':'synthetic_ROS_transport_not_simulation','passed':seen['nonempty_path_count']>0 and seen['commands']>0,'robot_execution_tested':False,**seen}
        Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if result['passed'] else 2
    finally:
        for c in children:
            if c.poll() is None:c.send_signal(signal.SIGINT)
        for c in children:
            try:c.wait(timeout=5)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        node.destroy_node();rclpy.shutdown()
if __name__=='__main__':raise SystemExit(main())
