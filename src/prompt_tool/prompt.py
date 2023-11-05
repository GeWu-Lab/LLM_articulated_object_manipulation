import json
from deoxys.utils.transform_utils import quat2mat,axisangle2quat,euler2mat
class Prompt:
    def __init__(self, content = None):
        if content is not None:
            self.content = content
        else:
            self.content = []

    def append_request(self, content):
        self.content.append(content)

    def append_response(self, response):
        self.content.append(response)

    def get_content(self):
        return self.content
    def append_environment_info(self, info):
        raise NotImplementedError
    
    def submit(self):
        raise NotImplementedError





class ManiPrompt(Prompt):
    
    def __init__(self, config_path,example_path, task = None):
        super().__init__()
        
        with open(config_path, "r") as f:
            content = json.load(f)
        with open(example_path, "r") as f:
            examples = json.load(f)
        
        example_content = ""
        for idx,example_key in enumerate(examples):
            example_task_name = examples[example_key]["task"]
            example_path = examples[example_key]["path"]
            example_description = examples[example_key]["description"]
            with open(example_path, "r") as f:

                records = json.load(f)
                scene_info = records["scene"]
                robot_state = scene_info["robot_state"]
                observations = scene_info["observations"]
                
                example_content += "Example {}, the task is: {}\n".format(idx, example_task_name)
                
                # example_content += "The franka initial infomation is as follows:\n"
                # example_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
                # example_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
                gripper_pos  = robot_state["gripper_pos"]
                gripper_quat = robot_state["gripper_quat"]
                gripper_pos = [round(x,3) for x in gripper_pos]
                gripper_quat = [round(x,3) for x in gripper_quat]
                example_content += "The xyz of gripper position is: {}\n".format(gripper_pos)
                # example_content += "The gripper quaternion is: {}\n".format(gripper_quat)
                example_content += "\n"
                
                
                example_content += "The URDF of object is as follows:\n"
                # example_content += "The object position is: {}\n".format(observations["object_pos"])
                # example_content += "The object quaternion is: {}\n".format(observations["object_quat"])
                handle_pos = observations["handle_pos"]
                joint_pos = observations["joint_pos"]
                handle_pos = [round(x,3) for x in handle_pos]
                joint_pos = [round(x,3) for x in joint_pos]
                
                # example_content += "The position of the articulation is: {}\n".format(joint_pos)
                # example_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
                # # example_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
                # # example_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
                joint_quat = observations["joint_quat"]
                joint_mat = quat2mat(joint_quat)
                axis = observations["urdf_conclusion"]["axis"]
                true_joint_axis = joint_mat.dot(axis)
                true_joint_axis = [round(x,3) for x in true_joint_axis]
                # example_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
                
                urdf_joint = "<joint name=\"joint_0\" type=\"{}\">\n<origin xyz=\"{} {} {}\"/>\n<axis xyz=\"{} {} {}\"/> \n<child link=\"movable_part\"/> \n<parent link=\"base_link\"/>\n</joint>\n"
                urdf_info = urdf_joint.format(observations["urdf_conclusion"]["joint_type"], \
                    joint_pos[0], joint_pos[1], joint_pos[2], true_joint_axis[0], true_joint_axis[1], true_joint_axis[2])
                
                example_content += urdf_info
                
                example_content += "the xyz position of movable_part manipulation point is: {}\n".format(handle_pos)
                
                # example_content += "The axis of the articulation is: {}\n".format(observations["urdf_conclusion"]["axis"])
                example_content += "\n"
                
                example_content += "The actions are as follows:\n"
                example_content += example_description
                
                example_content += "# According to my thought, the action sequences is: \n"
                actions = self.parse(scene_info["actions"])
                example_content += actions + "\n"

        
        
        system_prompt = content["system"]
        self.action_list = content["action_list"]
        system_prompt += "You could use the following action list to finish the task: \n"
        for action_name in self.action_list.keys():
            system_prompt += "{}: {}\n".format(action_name, self.action_list[action_name])
        
        self.append_request({"role": "system", "content": system_prompt})
        
        task_format = content["format"]
        self.init_content = ""
        self.init_content += "You should strictly follow the following answer format:\n" + task_format + "\n"
        self.init_content += "Here are some examples you can refer to:\n"
        self.init_content += example_content + "\n"
        if task is None:
            self.task = content["task"]
        else:
            task = task.replace("_", " ")
            self.task = task
            
        # self.init_content += "You should provide reasonable waypoints to deal with the articulated object, for example, generate the circle trajectories for revolute joint, the linear trajectories for prismatic joint\n"
        # self.init_content += "Think clearfully about the URDF information and articulated object structure to finish this task with the answer format: \n"
        
    def parse(self,actions):
        print("the actions are", actions)
        action_content = "[EXECUTE]\n"
        for idx,action in enumerate(actions):
            if action == "EXEC":
                action_content += "[EXEC]\n"
            elif action == "GRASP":
                action_content += "[GRASP]\n"
            elif action == "RELEASE":
                action_content += "[RELEASE]\n"
            elif action == "ANTICLOCKWISE_ROTATE":
                action_content += "[ANTICLOCKWISE_ROTATE]\n"
            elif action == "CLOCKWISE_ROTATE":
                action_content += "[CLOCKWISE_ROTATE]\n"
            else:
                action_pos = action[:3]
                action_pos = [round(x,3) for x in action_pos]
                action_content += "[MOVE] ({},{},{})\n".format(action_pos[0],action_pos[1],action_pos[2])
        action_content += "[END]\n"
        return action_content
        
    def append_environment_info(self, info):
        
        self.init_content += "You should think step by step to finish this task, think clearfully about the URDF information and object propertity to finish this task\n"
        self.init_content += "You could use the following action list and parameters defination to finish the task: \n"
        for action_name in self.action_list.keys():
            self.init_content += "{}: {}\n".format(action_name, self.action_list[action_name])
        
        self.init_content += "Rule: When we say anti-clockwise rotation around the z-axis, it refers to the results when viewed from the positive direction of the world coordinate system's z-axis. The other experssions of rotation follow this rule.\n"
        self.init_content += "The world coordinate system is defined as follows:\n"
        self.init_content += "The x-axis direction is [1,0,0], the y-axis direction is [0,1,0], the z-axis direction is [0,0,1]\n"
        
        
        self.init_content += "The task is: " + self.task + "\n"
        robot_state = info["robot_state"]
        observations = info["observations"]
        
        
        # self.init_content += "The franka initial infomation is as follows:\n"
        gripper_pos  = robot_state["gripper_pos"]
        gripper_quat = robot_state["gripper_quat"]
        gripper_pos = [round(x,3) for x in gripper_pos]
        gripper_quat = [round(x,3) for x in gripper_quat]
        # self.init_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
        # self.init_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
        self.init_content += "The xyz of gripper position is: {}\n".format(gripper_pos)
        # self.init_content += "The gripper quaternion is: {}\n".format(gripper_quat)
        self.init_content += "\n"
        
        
        self.init_content += "The URDF of object is as follows:\n"
        handle_pos = observations["handle_pos"]
        joint_pos = observations["joint_pos"]
        handle_pos = [round(x,3) for x in handle_pos]
        joint_pos = [round(x,3) for x in joint_pos]
        # self.init_content += "The object position is: {}\n".format(observations["object_pos"])
        # self.init_content += "The object quaternion is: {}\n".format(observations["object_quat"])
        # self.init_content += "the position of reference for manipulation is: {}\n".format(handle_pos)
        # # self.init_content += "The quaternion of the articulation is: {}\n".format(observations["joint_quat"])
        # self.init_content += "The position of the articulation is: {}\n".format(joint_pos)
        # self.init_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
        # self.init_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
        # self.init_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
        
        joint_quat = observations["joint_quat"]
        joint_mat = quat2mat(joint_quat)
        axis = observations["urdf_conclusion"]["axis"]
        true_joint_axis = joint_mat.dot(axis)
        true_joint_axis = [round(x,3) for x in true_joint_axis]
        # self.init_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
        
        urdf_joint = "<joint name=\"joint_0\" type=\"{}\">\n<origin xyz=\"{} {} {}\"/>\n<axis xyz=\"{} {} {}\"/> \n<child link=\"movable_part\"/> \n<parent link=\"base_link\"/>\n</joint>\n"
        urdf_info = urdf_joint.format(observations["urdf_conclusion"]["joint_type"], \
            joint_pos[0], joint_pos[1], joint_pos[2], true_joint_axis[0], true_joint_axis[1], true_joint_axis[2])
        
        self.init_content += urdf_info
        
        self.init_content += "the xyz position of movable_part manipulation point is: {}\n".format(handle_pos)
        self.init_content += "\n"
        # self.init_content += "I should think step by step, carefully consider the relationship between the joint and handle to finish this task\n"
        self.init_content += "The actions are as follows:\n"
        
    def submit(self):
        self.content.append({
            "role": "user",
            "content": self.init_content
        })


class ManiWithoutCOTPrompt(Prompt):
    
    def __init__(self, config_path,example_path,task = None):
        super().__init__()
        
        with open(config_path, "r") as f:
            content = json.load(f)
        with open(example_path, "r") as f:
            examples = json.load(f)
        
        example_content = ""
        for idx,example_key in enumerate(examples):
            example_task_name = examples[example_key]["task"]
            example_path = examples[example_key]["path"]
            example_description = examples[example_key]["description"]
            with open(example_path, "r") as f:

                records = json.load(f)
                scene_info = records["scene"]
                robot_state = scene_info["robot_state"]
                observations = scene_info["observations"]
                
                example_content += "Example {}, the task is: {}\n".format(idx, example_task_name)
                
                # example_content += "The franka initial infomation is as follows:\n"
                # example_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
                # example_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
                gripper_pos  = robot_state["gripper_pos"]
                gripper_quat = robot_state["gripper_quat"]
                gripper_pos = [round(x,3) for x in gripper_pos]
                gripper_quat = [round(x,3) for x in gripper_quat]
                example_content += "The xyz of gripper position is: {}\n".format(gripper_pos)
                # example_content += "The gripper quaternion is: {}\n".format(gripper_quat)
                example_content += "\n"
                
                
                example_content += "The URDF of object is as follows:\n"
                # example_content += "The object position is: {}\n".format(observations["object_pos"])
                # example_content += "The object quaternion is: {}\n".format(observations["object_quat"])
                handle_pos = observations["handle_pos"]
                joint_pos = observations["joint_pos"]
                handle_pos = [round(x,3) for x in handle_pos]
                joint_pos = [round(x,3) for x in joint_pos]
                
                # example_content += "The position of the articulation is: {}\n".format(joint_pos)
                # example_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
                # # example_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
                # # example_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
                joint_quat = observations["joint_quat"]
                joint_mat = quat2mat(joint_quat)
                axis = observations["urdf_conclusion"]["axis"]
                true_joint_axis = joint_mat.dot(axis)
                true_joint_axis = [round(x,3) for x in true_joint_axis]
                # example_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
                
                urdf_joint = "<joint name=\"joint_0\" type=\"{}\">\n<origin xyz=\"{} {} {}\"/>\n<axis xyz=\"{} {} {}\"/> \n<child link=\"movable_part\"/> \n<parent link=\"base_link\"/>\n</joint>\n"
                urdf_info = urdf_joint.format(observations["urdf_conclusion"]["joint_type"], \
                    joint_pos[0], joint_pos[1], joint_pos[2], true_joint_axis[0], true_joint_axis[1], true_joint_axis[2])
                
                example_content += urdf_info
                
                example_content += "the xyz position of movable_part manipulation point is: {}\n".format(handle_pos)
                
                # example_content += "The axis of the articulation is: {}\n".format(observations["urdf_conclusion"]["axis"])
                example_content += "\n"
                
                example_content += "The actions are as follows:\n"
                # example_content += example_description
                
                # example_content += "# According to my thought, the action sequences is: \n"
                actions = self.parse(scene_info["actions"])
                example_content += actions + "\n"

        
        
        system_prompt = content["system"]
        self.action_list = content["action_list"]
        system_prompt += "You could use the following action list to finish the task: \n"
        for action_name in self.action_list.keys():
            system_prompt += "{}: {}\n".format(action_name, self.action_list[action_name])
        
        self.append_request({"role": "system", "content": system_prompt})
        
        task_format = content["format"]
        self.init_content = ""
        self.init_content += "You should strictly follow the following answer format:\n" + task_format + "\n"
        self.init_content += "Here are some examples you can refer to:\n"
        self.init_content += example_content + "\n"
        
        if task is None:
            self.task = content["task"]
        else:
            task = task.replace("_", " ")
            self.task = task
        
        # self.init_content += "You should provide reasonable waypoints to deal with the articulated object, for example, generate the circle trajectories for revolute joint, the linear trajectories for prismatic joint\n"
        # self.init_content += "Think clearfully about the URDF information and articulated object structure to finish this task with the answer format: \n"
        
    def parse(self,actions):
        print("the actions are", actions)
        action_content = "[EXECUTE]\n"
        for idx,action in enumerate(actions):
            if action == "EXEC":
                action_content += "[EXEC]\n"
            elif action == "GRASP":
                action_content += "[GRASP]\n"
            elif action == "RELEASE":
                action_content += "[RELEASE]\n"
            elif action == "ANTICLOCKWISE_ROTATE":
                action_content += "[ANTICLOCKWISE_ROTATE]\n"
            elif action == "CLOCKWISE_ROTATE":
                action_content += "[CLOCKWISE_ROTATE]\n"
            else:
                action_pos = action[:3]
                action_pos = [round(x,3) for x in action_pos]
                action_content += "[MOVE] ({},{},{})\n".format(action_pos[0],action_pos[1],action_pos[2])
        action_content += "[END]\n"
        return action_content
        
    def append_environment_info(self, info):
        
        self.init_content += "You should think step by step to finish this task, think clearfully about the URDF information and object propertity to finish this task\n"
        self.init_content += "You could use the following action list and parameters defination to finish the task: \n"
        for action_name in self.action_list.keys():
            self.init_content += "{}: {}\n".format(action_name, self.action_list[action_name])
        
        # self.init_content += "Rule: When we say anti-clockwise rotation around the z-axis, it refers to the results when viewed from the positive direction of the world coordinate system's z-axis. The other experssions of rotation follow this rule.\n"
        # self.init_content += "The world coordinate system is defined as follows:\n"
        # self.init_content += "The x-axis direction is [1,0,0], the y-axis direction is [0,1,0], the z-axis direction is [0,0,1]\n"
        
        
        self.init_content += "The task is: " + self.task + "\n"
        robot_state = info["robot_state"]
        observations = info["observations"]
        
        
        # self.init_content += "The franka initial infomation is as follows:\n"
        gripper_pos  = robot_state["gripper_pos"]
        gripper_quat = robot_state["gripper_quat"]
        gripper_pos = [round(x,3) for x in gripper_pos]
        gripper_quat = [round(x,3) for x in gripper_quat]
        # self.init_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
        # self.init_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
        self.init_content += "The xyz of gripper position is: {}\n".format(gripper_pos)
        # self.init_content += "The gripper quaternion is: {}\n".format(gripper_quat)
        self.init_content += "\n"
        
        
        self.init_content += "The URDF of object is as follows:\n"
        handle_pos = observations["handle_pos"]
        joint_pos = observations["joint_pos"]
        handle_pos = [round(x,3) for x in handle_pos]
        joint_pos = [round(x,3) for x in joint_pos]
        # self.init_content += "The object position is: {}\n".format(observations["object_pos"])
        # self.init_content += "The object quaternion is: {}\n".format(observations["object_quat"])
        # self.init_content += "the position of reference for manipulation is: {}\n".format(handle_pos)
        # # self.init_content += "The quaternion of the articulation is: {}\n".format(observations["joint_quat"])
        # self.init_content += "The position of the articulation is: {}\n".format(joint_pos)
        # self.init_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
        # self.init_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
        # self.init_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
        
        joint_quat = observations["joint_quat"]
        joint_mat = quat2mat(joint_quat)
        axis = observations["urdf_conclusion"]["axis"]
        true_joint_axis = joint_mat.dot(axis)
        true_joint_axis = [round(x,3) for x in true_joint_axis]
        # self.init_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
        
        urdf_joint = "<joint name=\"joint_0\" type=\"{}\">\n<origin xyz=\"{} {} {}\"/>\n<axis xyz=\"{} {} {}\"/> \n<child link=\"movable_part\"/> \n<parent link=\"base_link\"/>\n</joint>\n"
        urdf_info = urdf_joint.format(observations["urdf_conclusion"]["joint_type"], \
            joint_pos[0], joint_pos[1], joint_pos[2], true_joint_axis[0], true_joint_axis[1], true_joint_axis[2])
        
        self.init_content += urdf_info
        
        self.init_content += "the xyz position of movable_part manipulation point is: {}\n".format(handle_pos)
        self.init_content += "\n"
        # self.init_content += "I should think step by step, carefully consider the relationship between the joint and handle to finish this task\n"
        self.init_content += "The actions are as follows:\n"
        
    def submit(self):
        self.content.append({
            "role": "user",
            "content": self.init_content
        })


class RoCoPrompt(Prompt):
    
    def __init__(self, config_path,example_path):
        super().__init__()
        
        with open(config_path, "r") as f:
            content = json.load(f)
        with open(example_path, "r") as f:
            examples = json.load(f)
        
        example_content = ""
        for idx,example_key in enumerate(examples):
            example_task_name = examples[example_key]["task"]
            example_path = examples[example_key]["path"]
            example_description = examples[example_key]["description"]
            with open(example_path, "r") as f:

                records = json.load(f)
                scene_info = records["scene"]
                robot_state = scene_info["robot_state"]
                observations = scene_info["observations"]
                
                example_content += "Example {}, the task is: {}\n".format(idx, example_task_name)
                
                example_content += "The franka initial infomation is as follows:\n"
                # example_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
                # example_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
                gripper_pos  = robot_state["gripper_pos"]
                gripper_quat = robot_state["gripper_quat"]
                gripper_pos = [round(x,3) for x in gripper_pos]
                gripper_quat = [round(x,3) for x in gripper_quat]
                example_content += "The gripper position is: {}\n".format(gripper_pos)
                # example_content += "The gripper quaternion is: {}\n".format(gripper_quat)
                example_content += "\n"
                
                
                example_content += "The object initial infomation is as follows:\n"
                # example_content += "The object position is: {}\n".format(observations["object_pos"])
                # example_content += "The object quaternion is: {}\n".format(observations["object_quat"])
                handle_pos = observations["handle_pos"]
                joint_pos = observations["joint_pos"]
                handle_pos = [round(x,3) for x in handle_pos]
                joint_pos = [round(x,3) for x in joint_pos]
                example_content += "the position of reference for manipulation is: {}\n".format(handle_pos)
                # example_content += "The position of the articulation is: {}\n".format(joint_pos)
                # example_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
                # example_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
                # example_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
                # joint_quat = observations["joint_quat"]
                # joint_mat = quat2mat(joint_quat)
                # axis = observations["urdf_conclusion"]["axis"]
                # true_joint_axis = joint_mat.dot(axis)
                # true_joint_axis = [round(x,3) for x in true_joint_axis]
                # example_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
                
                # example_content += "The axis of the articulation is: {}\n".format(observations["urdf_conclusion"]["axis"])
                example_content += "\n"
                
                example_content += "The actions are as follows:\n"
                # example_content += example_description
                
                actions = self.parse(scene_info["actions"])
                example_content += actions + "\n"

        
        
        system_prompt = content["system"]
        action_list = content["action_list"]
        system_prompt += "You could use the following action list to finish the task: \n"
        for action_name in action_list.keys():
            system_prompt += "{}: {}\n".format(action_name, action_list[action_name])
        
        self.append_request({"role": "system", "content": system_prompt})
        
        task_format = content["format"]
        self.init_content = ""
        self.init_content += "You should strictly follow the following answer format:\n" + task_format + "\n"
        self.init_content += "Here are some examples you can refer to:\n"
        self.init_content += example_content + "\n"
        task = content["task"]
        
        # self.init_content += "You should provide reasonable waypoints to deal with the articulated object, for example, generate the circle trajectories for revolute joint, the linear trajectories for prismatic joint\n"
        self.init_content += "The task is: " + task + "\n"
        # self.init_content += "Think clearfully about the URDF information and  articulated object structure to finish this task with the answer format: \n"
        
    def parse(self,actions):
        print("the actions are", actions)
        action_content = "[EXECUTE]\n"
        for idx,action in enumerate(actions):
            if action == "EXEC":
                action_content += "[EXEC]\n"
            elif action == "GRASP":
                action_content += "[GRASP]\n"
            elif action == "RELEASE":
                action_content += "[RELEASE]\n"
            elif action == "ANTICLOCKWISE_ROTATE":
                action_content += "[ANTICLOCKWISE_ROTATE]\n"
            elif action == "CLOCKWISE_ROTATE":
                action_content += "[CLOCKWISE_ROTATE]\n"
            else:
                action_pos = action[:3]
                action_pos = [round(x,3) for x in action_pos]
                action_content += "[MOVE] ({},{},{})\n".format(action_pos[0],action_pos[1],action_pos[2])
        action_content += "[END]\n"
        return action_content
        
    def append_environment_info(self, info):
        
        robot_state = info["robot_state"]
        observations = info["observations"]
        
        
        self.init_content += "The franka initial infomation is as follows:\n"
        gripper_pos  = robot_state["gripper_pos"]
        gripper_quat = robot_state["gripper_quat"]
        gripper_pos = [round(x,3) for x in gripper_pos]
        gripper_quat = [round(x,3) for x in gripper_quat]
        # self.init_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
        # self.init_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
        self.init_content += "The gripper position is: {}\n".format(gripper_pos)
        # self.init_content += "The gripper quaternion is: {}\n".format(gripper_quat)
        self.init_content += "\n"
        
        
        self.init_content += "The object initial infomation is as follows:\n"
        handle_pos = observations["handle_pos"]
        joint_pos = observations["joint_pos"]
        handle_pos = [round(x,3) for x in handle_pos]
        joint_pos = [round(x,3) for x in joint_pos]
        # self.init_content += "The object position is: {}\n".format(observations["object_pos"])
        # self.init_content += "The object quaternion is: {}\n".format(observations["object_quat"])
        self.init_content += "the position of reference for manipulation is: {}\n".format(handle_pos)
        # self.init_content += "The quaternion of the articulation is: {}\n".format(observations["joint_quat"])
        # self.init_content += "The position of the articulation is: {}\n".format(joint_pos)
        # self.init_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
        # self.init_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
        # self.init_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
        
        # joint_quat = observations["joint_quat"]
        # joint_mat = quat2mat(joint_quat)
        # axis = observations["urdf_conclusion"]["axis"]
        # true_joint_axis = joint_mat.dot(axis)
        # true_joint_axis = [round(x,3) for x in true_joint_axis]
        # self.init_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
        self.init_content += "\n"
        
        self.init_content += "The actions are as follows:\n"
        
    def submit(self):
        self.content.append({
            "role": "user",
            "content": self.init_content
        })

class RulePrompt(Prompt):
    
    def __init__(self, config_path,example_path,task = None):
        super().__init__()
        
        with open(config_path, "r") as f:
            content = json.load(f)
        with open(example_path, "r") as f:
            examples = json.load(f)
        
        example_content = ""
        for idx,example_key in enumerate(examples):
            example_task_name = examples[example_key]["task"]
            example_path = examples[example_key]["path"]
            example_description = examples[example_key]["description"]
            example_policy = examples[example_key]["policy"]
            with open(example_path, "r") as f:

                records = json.load(f)
                scene_info = records["scene"]
                robot_state = scene_info["robot_state"]
                observations = scene_info["observations"]
                
                example_content += "Example {}, the task is: {}\n".format(idx, example_task_name)
                
                example_content += "The franka initial infomation is as follows:\n"
                # example_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
                # example_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
                gripper_pos  = robot_state["gripper_pos"]
                gripper_quat = robot_state["gripper_quat"]
                gripper_pos = [round(x,3) for x in gripper_pos]
                gripper_quat = [round(x,3) for x in gripper_quat]
                example_content += "The gripper position is: {}\n".format(gripper_pos)
                # example_content += "The gripper quaternion is: {}\n".format(gripper_quat)
                example_content += "\n"
                
                
                example_content += "The object initial infomation is as follows:\n"
                # example_content += "The object position is: {}\n".format(observations["object_pos"])
                # example_content += "The object quaternion is: {}\n".format(observations["object_quat"])
                handle_pos = observations["handle_pos"]
                joint_pos = observations["joint_pos"]
                handle_pos = [round(x,3) for x in handle_pos]
                joint_pos = [round(x,3) for x in joint_pos]
                example_content += "the position of reference for manipulation is: {}\n".format(handle_pos)
                example_content += "The position of the articulation is: {}\n".format(joint_pos)
                example_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
                # example_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
                # example_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
                joint_quat = observations["joint_quat"]
                joint_mat = quat2mat(joint_quat)
                axis = observations["urdf_conclusion"]["axis"]
                true_joint_axis = joint_mat.dot(axis)
                true_joint_axis = [round(x,3) for x in true_joint_axis]
                example_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
                
                # example_content += "The axis of the articulation is: {}\n".format(observations["urdf_conclusion"]["axis"])
                example_content += "\n"
                
                example_content += "The actions are as follows:\n"
                example_content += example_description
                
                actions = self.parse(example_policy)
                example_content += actions + "\n"

        
        
        system_prompt = content["system"]
        action_list = content["action_list"]
        system_prompt += "You could use the following action list to finish the task: \n"
        for action_name in action_list.keys():
            system_prompt += "{}: {}\n".format(action_name, action_list[action_name])
        
        self.append_request({"role": "system", "content": system_prompt})
        
        task_format = content["format"]
        self.init_content = ""
        self.init_content += "You should strictly follow the following answer format:\n" + task_format + "\n"
        self.init_content += "Here are some examples you can refer to:\n"
        self.init_content += example_content + "\n"
        if task is None:
            task = content["task"]
        else:
            task = task.replace("_"," ")
        self.task = task
        # self.init_content += "You should provide reasonable waypoints to deal with the articulated object, for example, generate the circle trajectories for revolute joint, the linear trajectories for prismatic joint\n"
        self.init_content += "The task is: " + task + "\n"
        self.init_content += "Think clearfully about the URDF information and  articulated object structure to finish this task with the answer format: \n"
        
    def parse(self,actions):
        print("the actions are", actions)
        action_content = "[EXECUTE]\n"
        action_content += actions + "\n"
        action_content += "[END]\n"
        return action_content
        
    def append_environment_info(self, info):
        
        robot_state = info["robot_state"]
        observations = info["observations"]
        
        
        self.init_content += "The franka initial infomation is as follows:\n"
        gripper_pos  = robot_state["gripper_pos"]
        gripper_quat = robot_state["gripper_quat"]
        gripper_pos = [round(x,3) for x in gripper_pos]
        gripper_quat = [round(x,3) for x in gripper_quat]
        # self.init_content += "The franka position is: {}\n".format(robot_state["franka_pos"])
        # self.init_content += "The franka quaternion is: {}\n".format(robot_state["franka_quat"])
        self.init_content += "The gripper position is: {}\n".format(gripper_pos)
        # self.init_content += "The gripper quaternion is: {}\n".format(gripper_quat)
        self.init_content += "\n"
        
        
        self.init_content += "The object initial infomation is as follows:\n"
        handle_pos = observations["handle_pos"]
        joint_pos = observations["joint_pos"]
        handle_pos = [round(x,3) for x in handle_pos]
        joint_pos = [round(x,3) for x in joint_pos]
        # self.init_content += "The object position is: {}\n".format(observations["object_pos"])
        # self.init_content += "The object quaternion is: {}\n".format(observations["object_quat"])
        self.init_content += "the position of reference for manipulation is: {}\n".format(handle_pos)
        # self.init_content += "The quaternion of the articulation is: {}\n".format(observations["joint_quat"])
        self.init_content += "The position of the articulation is: {}\n".format(joint_pos)
        self.init_content += "The type of the articulation is: {}\n".format(observations["urdf_conclusion"]["joint_type"])
        # self.init_content += "The limit of the articulation is: {}\n".format(observations["urdf_conclusion"]["limit"])
        # self.init_content += "The current joint of the articulation is: {}\n".format(observations["extra_info"]["temp_dof"])
        
        joint_quat = observations["joint_quat"]
        joint_mat = quat2mat(joint_quat)
        axis = observations["urdf_conclusion"]["axis"]
        true_joint_axis = joint_mat.dot(axis)
        true_joint_axis = [round(x,3) for x in true_joint_axis]
        self.init_content += "The axis of the articulation is: {}\n".format(true_joint_axis)
        self.init_content += "\n"
        
        self.init_content += "The actions are as follows:\n"
        
    def submit(self):
        self.content.append({
            "role": "user",
            "content": self.init_content
        })

class CabinetPrompt(Prompt):
    
    def __init__(self, config_path,example_path):
        super().__init__()
        
        with open(config_path, "r") as f:
            content = json.load(f)
        with open(example_path, "r") as f:
            examples = f.read()
        
        system_prompt = content["system"]
        self.append_request({"role": "system", "content": system_prompt})
        task = content["task"]
        task_format = content["format"]
        self.init_content = ""
        self.init_content += "The task is: " + task + "\n"
        self.init_content += "You should strictly follow the following answer format:\n" + task_format + "\n"
        self.init_content += "Here are some examples you can refer to:\n"
        self.init_content += examples + "\n"
        
        self.init_content += "Please finish this task with the answer format: \n"
        
    def append_environment_info(self, info):
        self.init_content += "The environment information is:\n"
        gripper_state = info["gripper"]
        handle_state = info["handle"]
        revolute_door_shaft = info["shaft"]
        shaft_door_axis = info["shaft_axis"]
        gravity = info["gravity"]
        
        for key in info:
            if key not in ["gravity", "shaft_axis","handle"]:
                self.init_content += "the " + key + " position is: " + str(info[key]) + "\n"
            elif key in ["handle"]:
                self.init_content += "the position of the middle of the door {} is: ".format(key) + str(info[key]) + "\n"
            else:
                self.init_content += "the " + key + " direction is: " + str(info[key]) + "\n"
        
        self.init_content += "The actions are as follows:\n"
        
    def submit(self):
        self.content.append({
            "role": "user",
            "content": self.init_content
        })