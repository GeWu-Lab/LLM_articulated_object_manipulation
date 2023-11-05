from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply,quat_rotate

import math
import numpy as np
import torch
import random
import time
import json
import yaml
import os
import cv2
import moveit_msgs.msg


import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

USE_HUMAN = True


from urdfpy import URDF

def quat_axis(q, axis = 0):
    basic_vec = torch.zeros(q.shape[0],3, device=q.device)
    basic_vec[:,axis] = 1
    return quat_rotate(q, basic_vec)

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

object_SEG_ID = 1


def record(target_pos, target_quat):
    waypoint = np.zeros(7)
    waypoint[:3] = target_pos[0,:3]
    waypoint[3:] = target_quat[0]
    
    return waypoint

def init_sim(gym, args):
    
    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    # sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.use_gpu_pipeline = False
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        # sim_params.physx.enable_stabilization = True
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.friction_offset_threshold = 0.04
        sim_params.physx.friction_correlation_distance = 0.025
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    else:
        raise Exception("This example can only be used with PhysX")
    
    # create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    return sim

def load_object_asset(gym,sim, asset_root, path):
    object_asset_options = gymapi.AssetOptions()
    object_asset_options.density = 100
    object_asset_options.fix_base_link = True
    object_asset_options.disable_gravity = True
    object_asset_options.collapse_fixed_joints = True
    object_asset_options.use_mesh_materials = True
    object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    object_asset_options.override_com = True
    object_asset_options.override_inertia = True
    object_asset_options.armature = 0.001
    object_asset_options.vhacd_enabled = True
    object_asset_options.vhacd_params = gymapi.VhacdParams()
    object_asset_options.vhacd_params.resolution = 1000
    # object_asset_options.thickness = 0.0001
    print("object asset options are:", object_asset_options.thickness)
    object_asset = gym.load_asset(sim, asset_root, path, object_asset_options)
    
    return object_asset


def set_object_asset(gym, env,object_asset,pose, name, SCALE):
    object_handle = gym.create_actor(env, object_asset, pose,name, \
            segmentationId=object_SEG_ID)
    object_dof_props = gym.get_asset_dof_properties(object_asset)
    object_dof_props['stiffness'][0] = 1
    object_dof_props['friction'][0] = 0.1
    object_dof_props["effort"][0] = 100
    object_dof_props["driveMode"][0] = gymapi.DOF_MODE_POS
    gym.set_actor_dof_properties(env, object_handle, object_dof_props)
    gym.set_actor_scale(env, object_handle, SCALE)
    return object_handle

def load_object_pose(object_pose_path):
    if os.path.exists(object_pose_path):
        obj_info = yaml.safe_load(open(object_pose_path, "r"))
        obj_pos = obj_info["pos"]
        obj_ori = obj_info["ori"]
        obj_scale = obj_info["scale"]
    else:
        obj_pos = [0.5,0,0.2]
        obj_ori = [0,0,0,1]
        obj_scale = 0.2
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(obj_pos[0], obj_pos[1], obj_pos[2])
    obj_pose.r = gymapi.Quat(obj_ori[0], obj_ori[1], obj_ori[2], obj_ori[3])
    return obj_scale, obj_pose

def load_handle_pose(handle_path, scale):
    with open(handle_path, "r") as f:
        handle_info = yaml.safe_load(f)
    has_handle = handle_info["has_handle"]
    origin_handle_pose = np.array([handle_info["pos"]["x"], handle_info["pos"]["y"], handle_info["pos"]["z"]])
    gripper_ori = handle_info["gripper"]["ori"]
    
    handle_pose = torch.from_numpy(origin_handle_pose).float() * scale
    return has_handle, origin_handle_pose, handle_pose, gripper_ori
    
def load_franka_actor(gym,sim, asset_root, path):
    franka_asset_options = gymapi.AssetOptions()
    franka_asset_options.fix_base_link = True
    franka_asset_options.disable_gravity = True
    franka_asset_options.flip_visual_attachments = True
    franka_asset_options.armature = 0.01
    franka_asset = gym.load_asset(sim, asset_root, path, franka_asset_options)
    
    return franka_asset

def set_franka_actor(gym, env, franka_asset, franka_pose, name, gripper_ori = "x"):
    
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, name)
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    if gripper_ori == "x":
        reset_joint_positions = [0.03965490773843046, -1.175245273783704, -0.05830916477359954, -2.5399738537671346, -0.07468146039459143, 2.948025823473929, 0.8247538298964501]
    elif gripper_ori == "y":
        reset_joint_positions = [0.05876793904840425, -1.1162911515290097, -0.0758181008267759, -2.566077508997447, -0.07399603448973761, 2.9771455243742473, 2.4163162492067434]
    elif gripper_ori == "z":
        reset_joint_positions = [-2.2381279124612666e-05, 0.000816764082269449, -8.085571215387431e-05, -2.0001285138935776, 0.0005730844558349599, 1.999844442844003, 0.7901091570258141]
    
    
    joint_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"]

    # arm
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(600.0)
    franka_dof_props["damping"][:7].fill(40.0)

    # grippers
    franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][7:].fill(1000.0)
    franka_dof_props["damping"][7:].fill(0)

    # default dof states and position targets
    franka_num_dofs = gym.get_asset_dof_count(franka_asset)
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
    # default_dof_pos[:7] = franka_mids[:7]
    default_dof_pos[:7] = reset_joint_positions
    # grippers open
    default_dof_pos[7:] = franka_upper_limits[7:]

    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # send to torch
    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

    # get link index of panda hand, which we will use as end effector
    franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
    print("franka_link_dict: ", franka_link_dict)

    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    return franka_handle

def vis_pose(gym, viewer, env, pose):
    
    # vis
    axes_geom = gymutil.AxesGeometry(1)
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))
    
    gymutil.draw_lines(axes_geom, gym, viewer, env, pose)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, pose)

def update_obs(sim,envs,target_pos,target_quat,door_quat,handle_pos):
    
    for i in range(NUM_ENV):
        env = envs[i]

        handle_vis_pos = gymapi.Transform()
        handle_vis_pos.p = gymapi.Vec3(handle_pos[i][0], handle_pos[i][1], handle_pos[i][2])
        handle_vis_pos.r = gymapi.Quat(door_quat[i][0], door_quat[i][1], door_quat[i][2], door_quat[i][3])
        # vis_pose(gym, viewer, env, handle_vis_pos)

        door_vis_pos = gymapi.Transform()
        door_vis_pos.p = gymapi.Vec3(door_pos[i][0], door_pos[i][1], door_pos[i][2])
        door_vis_pos.r = gymapi.Quat(door_quat[i][0], door_quat[i][1], door_quat[i][2], door_quat[i][3])
        # vis_pose(gym, viewer, env, door_vis_pos)
        # print("the door pos is:", door_pos[i])
        
        gripper_vis_pos = gymapi.Transform()
        
        gripper_vis_pos.p = gymapi.Vec3(target_pos[i][0], target_pos[i][1], target_pos[i][2])
        gripper_vis_pos.r = gymapi.Quat(target_quat[i][0], target_quat[i][1], target_quat[i][2], target_quat[i][3])
        vis_pose(gym, viewer, env, gripper_vis_pos)


tt = 0
grasp = torch.zeros((1,1),dtype = bool,device="cuda:0")


def exec_grasp(gym,sim,dofs_pos,grasp_flag):
    grip_acts = torch.where(grasp_flag, torch.tensor([0,0], device=device), torch.tensor([0.04,0.04], device=device))
    pos_actions = torch.zeros_like(dofs_pos)
    pos_actions[:,:7] = dofs_pos[:,:7]
    pos_actions[:,7:9] = grip_acts
    pos_actions[:,9] = dofs_pos[:,9]
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_actions))

def plan_traj(dofs_pos,waypoint,motion_gen):
    tmp_dof_pos = list(dofs_pos[0,:7].cpu().numpy())
    joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    
    q_start = JointState.from_position(
        tensor_args.to_device([tmp_dof_pos]),
        joint_names=joint_names
    )
    quat_idx = [4,5,6,3]
    
    goal_pose = Pose(
        position=tensor_args.to_device([waypoint[ :3]]),
        quaternion=tensor_args.to_device([waypoint[quat_idx]]),
    )
    print("the goal pose is:", goal_pose)
    
    result = motion_gen.plan_single(q_start, goal_pose)
    
    while bool(result.success.cpu()[0]) is False:
        result = motion_gen.plan_single(q_start, goal_pose)
    #     print("result is:", result.success)
    # print("succ")
    # print("the result is:", result)
    interpolated_plan = result.interpolated_plan
    planned_joints_vel = interpolated_plan.velocity
    planned_joints_pos = interpolated_plan.position
    
    planned_joints_pos = planned_joints_pos.flip(dims = [0])
    planned_joints_vel = planned_joints_vel.flip(dims = [0])
    
    print("pos is", planned_joints_pos.shape)
    
    planned_time = interpolated_plan.shape[0]
    print("the interpolated_plan shape is:", interpolated_plan.shape)
    waypoints = np.zeros((5,7))
    way_idx = 0
    # save_json["scene"]["actions"].append("EXEC")
    return planned_time, planned_joints_pos, planned_joints_vel
                            
                            
def exec_plan(gym,sim,dofs_pos,dofs_vel, planned_joints_pos, planned_joints_vel,plan_time,exec_grasp_flag):
    # print("planned joint pos is:", planned_joints_pos[plan_time])
    # print("planned joint vel is:", planned_joints_vel[plan_time])
    
    grip_acts = torch.where(exec_grasp_flag, torch.tensor([0,0], device=device), torch.tensor([0.04,0.04], device=device))
    pos_actions = torch.zeros_like(dofs_pos)
    
    pos_actions[:,:7] = planned_joints_pos[plan_time]
    pos_actions[:,7:9] = grip_acts
    pos_actions[:,9:] = dofs_pos[:,9:]
    pos_actions = pos_actions
    
    vel_actions = torch.zeros_like(dofs_vel)
    vel_actions[:,:7] = planned_joints_vel[plan_time]
    vel_actions[:,7:] = dofs_vel[:,7:]
    vel_actions = vel_actions
    
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_actions))
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_actions))



if __name__ == "__main__":
    custom_parameters = [
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--index", "type": int, "default" : 0},
    {"name": "--task", "type": str, "default" : "close_door"},]
    args = gymutil.parse_arguments(
        description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
        custom_parameters=custom_parameters,
    )
    need_obj_idx = args.index
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    asset_root = "../assets"

    # acquire gym interface
    gym = gymapi.acquire_gym()
    # init sim
    sim = init_sim(gym, args)
    if sim is None:
        raise Exception("Failed to create sim")
    # init ground 
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)


    
    # Load the motion planner module
    
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    interpolation_dt = 0.05
    # create motion gen with a cuboid cache to be able to load obstacles later:
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        None,
        tensor_args,
        trajopt_tsteps=50,
        interpolation_steps=20000,
        num_ik_seeds=60,
        num_trajopt_seeds=6,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        grad_trajopt_iters=500,
        trajopt_dt=0.5,
        interpolation_dt=interpolation_dt,
        evaluate_interpolated_trajectory=True,
        js_trajopt_dt=0.5,
        js_trajopt_tsteps=34,
        velocity_scale=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5]
    )
    motion_gen = MotionGen(motion_gen_cfg)

    motion_gen.warmup(warmup_js_trajopt=False)

    CONFIG_PATH = "./task_config/{}.yaml".format(args.task)
    SCALE = 0.3
    LEN_OBJ = 1
    NUM_ENV = 1
    ENV_PER_ROW = int(math.sqrt(NUM_ENV))
    SPACING = 1.0
    ENV_LOWER = gymapi.Vec3(-SPACING, 0.0, -SPACING)
    ENV_UPPER = gymapi.Vec3(SPACING, SPACING, SPACING)
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    datasetPath = config["env"]["asset"]["datasetPath"]
    train_assets = config["env"]["asset"]["trainAssets"]
    
    envs = []
    object_assets = []
    object_poses = []
    object_handles = []
    object_load_handle_poses = torch.zeros((LEN_OBJ,3))
    object_handle_poses = []
    num_object_dofs = []
    object_rig_names = []

    train_assets_keys = list(train_assets.keys())
    all_assets = []
    #  create object asset
    for object_idx, name in enumerate(train_assets_keys):
        val = train_assets[name]
        candidate_object_name = name
        all_assets.append({"name": candidate_object_name, "val":val})
    
    object_name, val = all_assets[need_obj_idx]["name"], all_assets[need_obj_idx]["val"]
    object_name = object_name + "_{}".format(need_obj_idx)
    
    obj_path = os.path.join(datasetPath, val["path"])
    object_asset = load_object_asset(gym,sim, asset_root,obj_path )

    # load handle pos
    handle_pose_path = os.path.join(asset_root,datasetPath, val["handle"])
    
    object_pose_path = os.path.join(asset_root,datasetPath, val["pose"])
    
    SCALE, object_pose = load_object_pose(object_pose_path)
    has_hanlde, origin_handle_pose, object_handle_pose,gripper_ori = load_handle_pose(handle_pose_path, SCALE)
    object_load_handle_poses[0] = object_handle_pose
    

    # load door and handle pose
    object_poses.append(object_pose)
    object_assets.append(object_asset)
    
    object_rig_dict = gym.get_asset_rigid_body_dict(object_asset)
    print("the asset rig dict is:", object_rig_dict)
    object_rig_name = list(object_rig_dict.keys())[1]
    object_rig_names.append(object_rig_name)
    
    object_num_dofs = gym.get_asset_dof_count(object_asset)
    num_object_dofs.append(object_num_dofs)
        
    # create franka asset
    franka_path = "franka_description/robots/franka_panda.urdf"
    franka_asset = load_franka_actor(gym,sim, asset_root, franka_path)
    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(0, 0, 0)
    
    franka_handles = []

    object_door_indexs = []
    # create envs

    # Point camera at environments
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")
    
    cam_pos = gymapi.Vec3(0, 1, 1)
    cam_target = gymapi.Vec3(0.5, 0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # Create envs
    for i in range(NUM_ENV):
        env = gym.create_env(sim, ENV_LOWER, ENV_UPPER, NUM_ENV)
        # init franka in env
        franka_handle = set_franka_actor(gym, env, franka_asset, franka_pose, name = "franka",gripper_ori = gripper_ori)
        # gym.enable_actor_dof_force_sensors(env,franka_handle)
        
        franka_rigid_shape_props = gym.get_actor_rigid_shape_properties(env, franka_handle)
        # print("the franka rigid body props is:", len(franka_rigid_shape_props))
        for j in range(len(franka_rigid_shape_props)):
            print("the franka rigid body props is:", franka_rigid_shape_props[j].friction)
            franka_rigid_shape_props[j].friction = 1000
            franka_rigid_shape_props[j].rolling_friction = 1000
            # franka_rigid_shape_props[j].contact_offset = 1
            # print("the franka rigid body props is:", franka_rigid_shape_props[j].contact_offset)
        gym.set_actor_rigid_shape_properties(env,franka_handle , franka_rigid_shape_props)
        
        # init object in env
        target_name = "obj_{}".format(i)
        object_handle = set_object_asset(gym, env,object_assets[i],object_poses[i], target_name,SCALE)
        object_handles.append(object_handle)
        
        # get object rigid link
        object_rig_name = object_rig_names[i]
        object_rigid_body_index = gym.find_actor_rigid_body_index(env, object_handle, object_rig_name,gymapi.DOMAIN_ENV)
        object_door_indexs.append(object_rigid_body_index)
        
        # change the object rigid shape
        object_rigid_shape_props = gym.get_actor_rigid_shape_properties(env, object_handle)
        # print("the object rigid body props is:", len(object_rigid_shape_props))
        for j in range(len(object_rigid_shape_props)):
            # print("the object rigid body props is:", object_rigid_shape_props[j].friction)
            object_rigid_shape_props[j].friction = 1000.0
            object_rigid_shape_props[j].rolling_friction = 1000
        #     object_rigid_shape_props[j].contact_offset = 1
            # print("the object rigid body props is:", object_rigid_shape_props[j].contact_offset)
        gym.set_actor_rigid_shape_properties(env,object_handle , object_rigid_shape_props)
        
        dof_config_path = os.path.join(asset_root,datasetPath, val["jointValue"])
        if os.path.exists(dof_config_path):
            with open(dof_config_path, "r") as f:
                dof_config = yaml.safe_load(f)
                
                object_num_dofs = gym.get_asset_dof_count(object_assets[i])
                default_dof_pos = np.zeros(object_num_dofs, dtype=np.float32)
                default_dof_pos[0] = dof_config["dof"]
                print("the dof is:", dof_config["dof"])
                print("default dof pos is:", default_dof_pos)
                # init the object init dof
                # default_dof_pos = self._generate_init_object_dof()
                object_default_dof_pos = np.zeros(object_num_dofs, dtype=gymapi.DofState.dtype)
                object_default_dof_pos["pos"] = default_dof_pos
                # set initial dof states
                gym.set_actor_dof_states(env, object_handle, object_default_dof_pos, gymapi.STATE_ALL)
                # set initial position targets
                gym.set_actor_dof_position_targets(env, object_handle, default_dof_pos)
                
        envs.append(env)

        
    gym.prepare_sim(sim)

    # get link index of panda hand, which we will use as end effector
    franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
    print("the franka link are:", franka_link_dict)
    
    # franka_hand_index = franka_link_dict["panda_link8"]
    franka_ee_index = franka_link_dict["ee_link"]
    all_dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    dofs_pos = all_dof[:,0].view(NUM_ENV, 10)
    dofs_vel = all_dof[:,1].view(NUM_ENV, 10)

    all_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    all_state = all_state.view(NUM_ENV, -1,13)


    # step physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_mass_matrix_tensors(sim)
    
    # TODO: change to adapt all object not fixed index
    print("index is:", object_door_indexs)
    door_state = all_state[:, object_door_indexs[0],:]
    door_pos = door_state[:,:3]
    door_quat = door_state[:,3:7]

    ee_state = all_state[:,franka_ee_index,:]
    ee_pos = ee_state[:,:3]
    ee_quat = ee_state[:,3:7]

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    FRANKA_UPDATE_TIME = 0.1

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "move forward")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "move backward")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "move left")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "move right")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "move up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "move down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "grasp")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "release")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_I, "exec")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_H, "handle pos")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_N, "record data")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Z, "rotate_right")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "rotate_left")
    
    delta_pos = torch.zeros_like(dofs_pos).to(device)
    ee_down_dir = quat_axis(ee_quat, 2)
    
    waypoint = np.zeros(7)
    way_idx = 0
    
    exec_grasp_flag = torch.zeros((1,1),dtype = bool,device="cuda:0")
    grasp_time = 0
    
    planned_joints_pos = None
    planned_joints_vel = None
    planned_time = 0
    

    def parse_useful_urdf(path):
        object = URDF.load(object_path)
        urdf_info = {
            
        }
        for joint in object.actuated_joints:
            urdf_info["joint_name"] = joint.name
            urdf_info["pose_from_parent"] = joint.origin.reshape(-1).tolist()
            urdf_info["axis"] = joint.axis.tolist()
            urdf_info["limit"] = [joint.limit.lower, joint.limit.upper]
            urdf_info["joint_type"] = joint.joint_type
        return urdf_info


    handle_pos = quat_apply(door_quat, object_load_handle_poses) + door_pos
    object_path = os.path.join(asset_root,obj_path)

    
    save_json = {"name": object_name, "scene":{
        "actions": [],
        "robot_state":{
            "franka_pos" : [franka_pose.p.x, franka_pose.p.y, franka_pose.p.z],
            "franka_quat" : [franka_pose.r.x, franka_pose.r.y, franka_pose.r.z, franka_pose.r.w],
            "gripper_pos": ee_pos[0].cpu().numpy().tolist(),
            "gripper_quat": ee_quat[0].cpu().numpy().tolist(),
        },
        "observations": {
            "object_pos": [object_pose.p.x, object_pose.p.y, object_pose.p.z],
            "object_quat": [object_pose.r.x, object_pose.r.y, object_pose.r.z, object_pose.r.w],
            "object_scale": SCALE,
            "handle_origin": origin_handle_pose.tolist(),
            "handle_pos": handle_pos[0].cpu().numpy().tolist(),
            "joint_pos": door_pos[0].cpu().numpy().tolist(),
            "joint_quat": door_quat[0].cpu().numpy().tolist(),
            "urdf_conclusion":parse_useful_urdf(object_path),
            "extra_info": {
                "temp_dof": dofs_pos[0,-1].cpu().numpy().tolist()
            }
        },
    }}
    
    from utils import quat2mat,mat2quat
    observations = save_json["scene"]["observations"]
    joint_quat = observations["joint_quat"]
    joint_mat = quat2mat(joint_quat)
    axis = observations["urdf_conclusion"]["axis"]
    true_joint_axis = joint_mat.dot(axis)
    save_json["scene"]["observations"]["true_axis"] = true_joint_axis.tolist()
    target_pos = ee_pos.clone()
    target_quat = ee_quat.clone()
    go_flag = True

    

    while not gym.query_viewer_has_closed(viewer) and go_flag:
        
        # step physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)
        handle_pos = quat_apply(door_quat, object_load_handle_poses) + door_pos
        # print("the handle pos is:", handle_pos)
        t = gym.get_sim_time(sim)
        if t >= FRANKA_UPDATE_TIME:
            FRANKA_UPDATE_TIME += 0.05
            
            if grasp_time > 0:
                exec_grasp(gym,sim,dofs_pos, exec_grasp_flag)
                grasp_time -= 1
            elif planned_time > 0:
                planned_time -= 1
                # print(planned_joints_pos)
                exec_plan(gym,sim,dofs_pos,dofs_vel, planned_joints_pos,planned_joints_vel, planned_time,exec_grasp_flag)
            else:
                evt = gym.query_viewer_action_events(viewer)
                for e in evt:
                    if  e.value > 0:
                        input_cmd = e.action
                        if input_cmd == "reset":
                            target_pos = ee_pos.clone()
                            target_quat = ee_quat.clone()
                            
                        if input_cmd == "grasp":
                            exec_grasp_flag = torch.ones((1,1),dtype = bool,device="cuda:0")
                            save_json["scene"]["actions"].append("GRASP")
                            grasp_time = 5
                        elif input_cmd == "release":
                            exec_grasp_flag = torch.zeros((1,1),dtype = bool,device="cuda:0")
                            save_json["scene"]["actions"].append("RELEASE")
                            grasp_time = 5
                        if input_cmd == "move forward":
                            target_pos[:,0] += 0.01
                        elif input_cmd == "move backward":
                            target_pos[:,0] += -0.01
                        # complete the command for different direction
                        elif input_cmd == "move left":  
                            target_pos[:,1] += 0.01
                        elif input_cmd == "move right":
                            target_pos[:,1] += -0.01
                        elif input_cmd == "move up":
                            target_pos[:,2] += 0.01
                        elif input_cmd == "move down":
                            target_pos[:,2] += -0.01
                        elif input_cmd == "handle pos":
                            target_pos[:,:3] = handle_pos 
                        elif input_cmd == "rotate_right":
                            hand_q = target_quat[0].cpu().numpy()
                            print("temp quat is:", hand_q)
                            move_quat = np.array([0.        , 0.        , 0.25881905, 0.96592583])
                            move_matrix = quat2mat(move_quat)
                            temp_matrix = quat2mat(hand_q)
                            
                            target_matrix = temp_matrix.dot(move_matrix)
                            target_q = mat2quat(target_matrix)
                            print("target q is:", target_q)
                            target_quat[:,:] = torch.from_numpy(target_q).float()
                            
                            waypoint = record(target_pos, target_quat)
                            planned_time, planned_joints_pos, planned_joints_vel = plan_traj(dofs_pos,waypoint,motion_gen)
                            save_json["scene"]["actions"].append("CLOCKWISE_ROTATE")
                            
                        elif input_cmd == "rotate_left":
                            hand_q = target_quat[0].cpu().numpy()
                            print("temp quat is:", hand_q)
                            move_quat = np.array([ 0.        ,  0.        , -0.25881905,  0.96592583])
                            move_matrix = quat2mat(move_quat)
                            temp_matrix = quat2mat(hand_q)
                            
                            target_matrix = temp_matrix.dot(move_matrix)
                            target_q = mat2quat(target_matrix)
                            print("target q is:", target_q)
                            target_quat[:,:] = torch.from_numpy(target_q).float()
                            
                            waypoint = record(target_pos, target_quat)
                            planned_time, planned_joints_pos, planned_joints_vel = plan_traj(dofs_pos,waypoint,motion_gen)
                            save_json["scene"]["actions"].append("ANTICLOCKWISE_ROTATE")
                        
                        if e.action == "record data":
                            target_p = target_pos[0,:3]

                            print("target quat is:", target_quat)
                            waypoint = record(target_pos, target_quat)
                            

                            print("the waypoints is:", waypoint)
                            planned_time, planned_joints_pos, planned_joints_vel = plan_traj(dofs_pos,waypoint,motion_gen)
                            save_json["scene"]["actions"].append(waypoint.tolist())
                            
                        if e.action == "save point":
                            json.dump(save_json, open("human_demonstration/{}.json".format(object_name), "w+"))
                            go_flag = False
                        # if e.action == "exec":
                        #     planned_time, planned_joints_pos, planned_joints_vel = plan_traj(dofs_pos,waypoint,motion_gen)


            gym.clear_lines(viewer)
            update_obs(sim, envs,target_pos,target_quat,door_quat,handle_pos)
            

        gym.step_graphics(sim)
        
        # step rendering
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)