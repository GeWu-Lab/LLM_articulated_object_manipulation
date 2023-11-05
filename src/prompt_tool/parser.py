
class Parser:
    
    def __init__(self) -> None:
        pass
    
    def parse(self, response):
        raise NotImplementedError


class ManiParser(Parser):
    def __init__(self) -> None:
        pass
    
    
    def parse_action(self, actions):
        parsed_actions = []
        try:
            content_list = actions.split("\n")
            ERROR = None
            ignored_ids = []
            for cmd_idx, e in enumerate(content_list):
                if e.startswith("#") or e.startswith(" "):
                    ignored_ids.append(cmd_idx)
            for ignore_id in ignored_ids:
                content_list.pop(ignore_id)
            print("content list is:", content_list)
            
            points = []
            if content_list[0] != "[EXECUTE]" or content_list[-1] != "[END]":
                ERROR = "The answer format is wrong"
                return ERROR, points
            
            proposed_actions = content_list[1:-1]
            

            for action in proposed_actions:
                if action.startswith("[MOVE]"):
                    parsed_pos = action[8:-1]
                    xyz = parsed_pos.split(",")
                    target_action = {
                        "type": "move",
                        "target": [float(x) for x in xyz]
                    }
                    parsed_actions.append(target_action)
                elif action.startswith("[EXEC]"):
                    parsed_actions.append({"type": "exec"})
                elif action.startswith("[GRASP]"):
                    # points.append(parsed_point)
                    parsed_actions.append({"type": "grasp"})
                elif action.startswith("[RELEASE]"):
                    parsed_actions.append({"type": "release"})
                elif action.startswith("[ANTICLOCKWISE_ROTATE]"):
                    parsed_actions.append({"type": "rotate_anti_clockwise"})
                elif action.startswith("[CLOCKWISE_ROTATE]"):
                    parsed_actions.append({"type": "rotate_clockwise"})
                    
            parsed_actions.append({"type": "end"})
            print("parsed actions: ", parsed_actions)
        except Exception as e:
            print(e)
            ERROR = "The proposed points are not in the correct format"
        finally:
            print("end parsing")
            return ERROR, parsed_actions
        
    def parse(self, response):
        res_content= response["choices"][0]["message"]["content"]
        print("response content is: ", res_content)
        parsed_actions = []
        try:
            content_list = res_content.split("\n")
            ERROR = None
            ignored_ids = []
            start_id = 0
            for cmd_idx, e in enumerate(content_list):
                if e.startswith("The most confident"):
                    start_id = cmd_idx
                    break
                # if e.startswith("#") or e.startswith(" ") or e.startswith("According"):\
                # if not e.startswith("["):
                #     ignored_ids.append(cmd_idx)
            content_list = content_list[start_id:]
            for cmd_idx, e in enumerate(content_list):
                if not e.startswith("["):
                    ignored_ids.append(cmd_idx)
            ignored_ids.reverse()
            for ignore_id in ignored_ids:
                content_list.pop(ignore_id)
            print("content list is:", content_list)

            points = []
            if content_list[0] != "[EXECUTE]" or content_list[-1] != "[END]":
                ERROR = "The answer format is wrong"
                return ERROR, points
            
            proposed_actions = content_list[1:-1]
            

            for action in proposed_actions:
                if action.startswith("[MOVE]"):
                    parsed_pos = action[8:-1]
                    xyz = parsed_pos.split(",")
                    
                    target_action = {
                        "type": "move",
                        "target": [float(x.strip()) for x in xyz]
                    }
                    parsed_actions.append(target_action)
                if action.startswith("[GRASP]"):
                    # points.append(parsed_point)
                    # parsed_actions.append({"type": "exec"})
                    parsed_actions.append({"type": "grasp"})
                elif action.startswith("[RELEASE]"):
                    # parsed_actions.append({"type": "exec"})
                    parsed_actions.append({"type": "release"})
                elif action.startswith("[ANTICLOCKWISE_ROTATE]"):
                    parsed_actions.append({"type": "rotate_anti_clockwise"})
                elif action.startswith("[CLOCKWISE_ROTATE]"):
                    parsed_actions.append({"type": "rotate_clockwise"})
                elif action.startswith("[EXEC]"):
                    parsed_actions.append({"type": "exec"})
            # parsed_actions.append({"type": "exec"})
            parsed_actions.append({"type": "end"})
            print("parsed actions: ", parsed_actions)
        except Exception as e:
            print(e)
            ERROR = "The proposed points are not in the correct format"
        finally:
            print("end parsing")
            return ERROR, parsed_actions



class RuleParser(Parser):
    def __init__(self) -> None:
        pass
    
    def parse_action(self, actions):
        raise NotImplementedError
        
    def parse(self, response):
        
        res_content= response["choices"][0]["message"]["content"]
        print("response content is: ", res_content)
        parsed_actions = []
        try:
            content_list = res_content.split("\n")
            ERROR = None
            for cmd_idx, e in enumerate(content_list):
                if e.startswith("#"):
                    content_list.pop(cmd_idx)

            points = []
            if content_list[0] != "[EXECUTE]" or content_list[-1] != "[END]":
                ERROR = "The answer format is wrong"
                return ERROR, points
            
            proposed_actions = content_list[1:-1]
            
            parsed_actions= proposed_actions[0]
            print("parsed actions: ", parsed_actions)
            
        except Exception as e:
            print(e)
            ERROR = "The proposed points are not in the correct format"
        finally:
            print("end parsing")
            return ERROR, parsed_actions




class CabinetParser(Parser):
    def __init__(self) -> None:
        pass
    
    def parse(self, response):
        res_content= response["choices"][0]["message"]["content"]
        print("response content is: ", res_content)
        try:
            content_list = res_content.split("\n")
            ERROR = None
            points = []
            if content_list[1] != "[EXECUTE]" or content_list[5] != "[GRASP]" or content_list[-1] != "[END]":
                ERROR = "The answer format is wrong"
                return ERROR, points
            
            proposed_points = content_list[2:-1]
            parsed_point = None
            for point in proposed_points:
                if point == "[GRASP]":
                    # points.append(parsed_point)
                    continue
                parsed_point = point[1:-1]
                parsed_point = parsed_point.split(",")
                parsed_point = [float(x) for x in parsed_point]
                points.append(parsed_point)
            
        except Exception as e:
            print(e)
            ERROR = "The proposed points are not in the correct format"
        finally:
            print("end parsing")
            return ERROR, points

if __name__ == "__main__":
    l = ManiParser()
    response = {
        "choices": [{"message":{"content":"parsing\n[EXECUTE]\n[MOVE] (0,0,0)\n[GRASP]\n[MOVE] (0,0,0)\n[RELEASE]\n[MOVE] (0,0,0)\n[END]"}}]
    }
    error, proposed_actions = l.parse(response)
    print(proposed_actions)