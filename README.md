# Visual-Robot-Project

Setup:
Enter the following into the terminal upon installing the files:
Step 1 - "python -m venv venv"

Step 2 - "venv/scripts/activate"

Step 3 - "pip install panda-gym stable_baselines3 opencv-python tensorboard"

Then it should work correctly

------------------------------------------------------------------

Notes for meeting:

Completed:
- Set up a working camera that sits at the first joint of the robot arm for now
- Recreated a basic flip task
- Tested some Image Segmentation
- Looked at dense rewards in PandaFlipDense-v3 briefly

To do:
(In the next two weeks):
- Figure out how to specify dense rewards to teach robot to move to the right place and pick it up, and then rotate correctly
- Decide on how the camera will function, i.e. What position and where it will look, as well as what method of visual perception I will need to use to identify beakers
- tensorboard

(In general):
- Create beaker URDFs and any other objects I might need (In Blender)
- Create a working method for training the robot via dense rewards (or something else if I find a better method)
- Train the robot to look at the camera data instead and be able to pick things up using that data instead
- Store a trained model
- Test the model on the new environment containing the beakers