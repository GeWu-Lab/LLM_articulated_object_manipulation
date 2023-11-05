# Kinematic-aware Prompting for Generalizable Articulated Object Manipulation with LLMs
[[Project Page](https://xwinks.github.io/llm_for_articulated_manipulation/)] [[Arxiv]()]


![image](./assets/pipeline.png)

# Introduction

In this work, we delve into the problem of harnessing LLMs for generalizable articulated object manipulation, recognizing that the rich world knowledge inherent in LLMs is adept at providing a reasonable manipulation understanding of various articulated objects.

# Install
In this work, we use [Isaac gym](https://developer.nvidia.com/isaac-gym) as the simulation environment, the [curobo](https://curobo.org/) as the motion planner. This code is tested in Ubuntu 20.04, pytorch 1.13.1+cu117, Isaac gym 2020.2.

First install the requirements:

```
pip install -r requirements.txt
```

Then install the Isaac gym and curobo according to their official documents.

# Demonstration Collection

User could first download the articulated objects from [here](https://drive.google.com/file/d/1iWoY4jmi-1mDt8Th907zNvfh0d3E9hL9/view?usp=drive_link). 

User could collect the human demonstration by running
```
python human_manipulation --task open_door --index 0
```
, the keyboard could be used to determine the next waypoint following the rule below as defined in the `subscribe_viewer_keyboard_event` function:
```
    W, "move forward"
    S, "move backward"
    A, "move left"
    D, "move right"
    Q, "move up"
    E, "move down"
    G, "grasp"
    V, "release"
    I, "exec"
    R, "reset"
    H, "move to handle pos"
    N, "record data"
    Z, "rotate_right"
    X, "rotate_left"
```
We have provided visualization for the target waypoint. Once the target waypoint is determined, user could press `N` to record the data and move the franka arm to the target waypoint. 
When the task if finished, user could press `L` to save the trajectory.

To replay the human demonstration, user could use the command below.
```
python replay_human_manipulation.py --demo_path open_drawer_19179_0
```


# Evaluation

To prompt GPT generate a reasonable trajectory, user should first change the openai key in `prompt_tool/agent.py`, then run

```
python gpt_manipulation.py --task open_drawer
```

We have provided part of the manipulation demonstrations in `prompt_config` and `rotate_records`. User could also follow the format to prompt GPT with own demonstration dataset.

