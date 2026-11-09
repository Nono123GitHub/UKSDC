# =====================================================================
# MINI GUIDE: CUSTOMISING YOUR LIMB (edit the CHEETAH_XML string)
# =====================================================================
#
# How a limb is built
#   A limb is a chain of nested <body> blocks: thigh > shin > foot.
#   Each <body> holds 3 things:
#     <joint>  the hinge it rotates around
#     <geom>   the visible, solid capsule (the "bone")
#     <body>   the next segment down the chain (optional)
#   Every joint also needs a <motor> in <actuator>, or it can't be driven.
#
# All angles are in RADIANS. Degrees to radians: degrees * 0.01745
#   (30 deg = 0.52, 45 deg = 0.785, 90 deg = 1.57)
#
# ---------------------------------------------------------------------
# 1. MOVE WHERE THE LIMB ATTACHES
#    <body name="fthigh" pos="x y z">
#      x: along the torso, -0.5 = back end, 0 = middle, 0.5 = front end
#      y: sideways, purely visual (the cheetah is 2D)
#      z: up/down, negative = hangs lower
#
# 2. CHANGE A SEGMENT'S LENGTH OR THICKNESS
#    <geom size="radius half_length">
#      Full length of the capsule = 2 * half_length
#      e.g. size="0.046 .2" gives a 0.4 long bone
#
# 3. CHANGE A SEGMENT'S RESTING ANGLE
#    <geom axisangle="0 1 0 a">
#      a = 0 points straight down, positive tilts one way, negative the other
#
# 4. KEEP THE BONE ATTACHED TO ITS JOINT (important!)
#    After changing half_length (h) or angle (a), fix the geom's pos:
#      geom pos = ( -h*sin(a), 0, -h*cos(a) )
#    and fix the NEXT body's pos so it starts at the end of this bone:
#      child body pos = ( -2*h*sin(a), 0, -2*h*cos(a) )
#    Quick helper:
#      import math
#      h, a = 0.133, 0.52
#      print(-h*math.sin(a), -h*math.cos(a))          # geom pos x, z
#      print(-2*h*math.sin(a), -2*h*math.cos(a))      # child body pos x, z
#
# 5. CHANGE HOW FAR A JOINT CAN BEND
#    <joint range="min max">    e.g. range="-1 .7"
#      Wider range = floppier, more flexible limb
#
# 6. CHANGE HOW SPRINGY OR SLUGGISH A JOINT FEELS
#    <joint stiffness="180" damping="4.5">
#      stiffness: spring pulling the joint back to its resting angle
#      damping:   resistance to moving fast (higher = smoother, slower)
#
# 7. CHANGE HOW STRONG A MOTOR IS
#    <motor gear="120" joint="fthigh">
#      torque = action (-1 to 1) * gear
#      Thighs are usually strongest, feet weakest
#
# 8. CHANGE THE COLOUR
#    <geom rgba="R G B A">    values 0 to 1, e.g. "0.4 0.6 0.9 1" = blue
#
# 9. ADD ANOTHER SEGMENT (e.g. a toe under the foot)
#    Nest a new <body> inside the foot's <body>, give it a <joint> and
#    a <geom>, then add a matching <motor> line in <actuator>.
#
# 10. ADD A WHOLE NEW LEG
#    Copy an entire leg block, paste it inside <body name="torso">,
#    change its pos, and RENAME EVERY name="..." so none repeat.
#    Then add one <motor> per new joint.
#
# 11. REMOVE A SEGMENT OR LEG
#    Delete its <body> block AND its <motor> lines, or MuJoCo will crash
#    with "unknown joint".
#
# 12. MOVE THE HEAD
#    The head is a single <geom name="head"> inside the torso body.
#    It has no joint, so it's welded on. Change its pos x to slide it
#    along the torso (currently .6 = just past the front end).
#
# ---------------------------------------------------------------------
# ACTION SPACE RULES
#   - Action size = number of <motor> lines (updates automatically in v5)
#   - Action order = order of the <motor> lines, top to bottom
#   - Update the docstring at the bottom if you add, remove or reorder
#
# COMMON ERRORS
#   "repeated name"      two things share a name, rename one
#   "unknown joint"      a <motor> points to a joint that doesn't exist
#   Limb floats apart    geom/body pos not updated (see step 4)
#   Cheetah flips a lot  motor gear too high or joint range too wide
#   Model won't load     you changed the body, retrain with --train
# =====================================================================
#
# =====================================================================
# MINI GUIDE: THE ANTI-FLIP REWARD (edit the settings below the imports)
# =====================================================================
#
# This is reinforcement learning (RL):
#   Environment = the standard 2-legged HalfCheetah simulation
#   Agent       = the SAC neural network that picks the 6 torques
#   Reward      = the score below, which the agent learns to maximise
#
# Every step the cheetah gets:
#   + forward * speed            run to the right
#   - control * torque^2         don't waste energy
#   + alive                      small bonus just for not having flipped
#   - upright * |tilt angle|     stay level
# And if |tilt angle| > FLIP_ANGLE (85 degrees):
#   - FLIP_PENALTY, and the episode ends immediately
#
# How it knows it flipped:
#   data.qpos[2] is the "rooty" joint, the whole torso's tilt in radians
#   (0 = level). It's wrapped to -pi..pi so spinning round once doesn't
#   read as 6.28, then compared against FLIP_ANGLE.
#
# Why "alive" matters: without it, a struggling cheetah earns negative
# reward each step and might LEARN to flip on purpose to end the
# episode early. The alive bonus makes staying upright always worth it.
#
# Tuning:
#   Still flips?           raise upright or FLIP_PENALTY, or lower FLIP_ANGLE
#   Upright but too slow?  raise forward (e.g. 2.0)
#   Jittery legs?          raise control
#
# When things stop:
#   Training  stops after TRAIN_STEPS, saves the model, then watching starts
#   Watching  runs until WATCH_EPISODES is reached or you press Ctrl+C
#   Episode   resets on a flip, or after 1000 steps (built-in time limit)
#
# Running:
#   python cheetah_2leg.py           train if no saved model, then watch
#   python cheetah_2leg.py --train   always retrain, then watch
#   Changing the reward or body means you must retrain (--train).
# =====================================================================

import os
import sys
import time
import tempfile
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

# ---------------------------------------------------------------------
# REWARD SETTINGS
# ---------------------------------------------------------------------
REWARD_WEIGHTS = {
    "height": 10.0,   # bonus per unit of torso height above JUMP_START_HEIGHT
    "rise": 1.0,      # bonus per unit of upward speed
    "airtime": 1.0,   # bonus every step the whole cheetah is off the floor
    "forward": 0.0,   # bonus per unit of forward speed (0 = jump on the spot)
    "control": 0.1,   # penalty for torque use
    "alive": 1.0,     # bonus every step it hasn't flipped
    "upright": 1.0,   # penalty per radian of tilt
}
JUMP_START_HEIGHT = 0.65      # torso height where a jump starts counting (resting is about 0.57)
FLIP_ANGLE = np.radians(85)   # 85 degrees; beyond this counts as flipped
FLIP_PENALTY = 100.0          # one-off penalty when it flips

# ---------------------------------------------------------------------
# TRAINING / WATCHING SETTINGS
# ---------------------------------------------------------------------
TRAIN_STEPS = 200_000          # more = better, but slower (roughly 1 to 2 hours on CPU)
MODEL_PATH = "cheetah_2leg_jump_sac"   # saved as cheetah_2leg_jump_sac.zip
WATCH_EPISODES = 10            # how many episodes to show before closing

# Standard 2-legged HalfCheetah body, written out so you can customise it with the guide above
CHEETAH_XML = """
<mujoco model="cheetah_2leg">
  <compiler angle="radian" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>

      <!-- Head: at the front of the torso (standard position) -->
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>

      <!-- Back leg -->
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>

      <!-- Front leg -->
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90"  joint="bshin"  name="bshin"/>
    <motor gear="60"  joint="bfoot"  name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60"  joint="fshin"  name="fshin"/>
    <motor gear="30"  joint="ffoot"  name="ffoot"/>
  </actuator>
</mujoco>
"""


class JumpReward(gym.Wrapper):
    """Replaces the built-in reward with one that rewards jumping and punishes flipping."""

    def __init__(self, env):
        super().__init__(env)
        # Look up the floor's id once, so we can spot floor contacts each step
        self.floor_id = self.env.unwrapped.model.geom("floor").id

    def is_airborne(self, data):
        # True if none of MuJoCo's current contacts involve the floor
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == self.floor_id or contact.geom2 == self.floor_id:
                return False
        return True

    def step(self, action):
        # Ignore the original reward (the "_") and build our own
        obs, _, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data
        w = REWARD_WEIGHTS

        torso_height = data.body("torso").xpos[2]                 # height above the floor
        up_speed = data.qvel[1]                                   # vertical speed (positive = up)
        tilt = (data.qpos[2] + np.pi) % (2 * np.pi) - np.pi       # tilt in radians, wrapped to -pi..pi
        airborne = self.is_airborne(data)

        reward = (
            w["height"] * max(0.0, torso_height - JUMP_START_HEIGHT)
            + w["rise"] * max(0.0, up_speed)
            + w["airtime"] * float(airborne)
            + w["forward"] * info["x_velocity"]
            - w["control"] * float(np.sum(np.square(action)))
            + w["alive"]
            - w["upright"] * abs(tilt)
        )

        flipped = abs(tilt) > FLIP_ANGLE
        if flipped:
            reward -= FLIP_PENALTY
            terminated = True   # end the episode, so flipping also loses all future reward

        info["tilt"] = tilt
        info["flipped"] = flipped
        return obs, reward, terminated, truncated, info


def write_model_file():
    # MuJoCo needs a real file path, so save the XML to a temp file
    path = os.path.join(tempfile.gettempdir(), "cheetah_2leg.xml")
    with open(path, "w") as f:
        f.write(CHEETAH_XML)
    return path


def make_env(render_mode=None):
    # v5 is required: it accepts xml_file and sizes the action/observation spaces from the model
    env = gym.make("HalfCheetah-v5", xml_file=write_model_file(), render_mode=render_mode)
    return JumpReward(env)


def train():
    print(f"Training 2-legged cheetah for {TRAIN_STEPS} steps (no window, this takes a while)...")
    env = make_env(render_mode=None)   # no rendering = much faster training
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    env.close()
    print(f"Saved trained model to {MODEL_PATH}.zip")


def watch():
    env = make_env(render_mode="human")
    model = SAC.load(MODEL_PATH, env=env)

    print(f"Action space: {env.action_space}")            # Box(-1, 1, (6,), float32)
    print(f"Observation space: {env.observation_space}")
    observation, info = env.reset(seed=42)
    episode_reward, episodes, flips = 0.0, 0, 0
    print("Watching trained cheetah. Press Ctrl+C in the terminal to exit.")

    try:
        while episodes < WATCH_EPISODES:
            # The trained policy picks the torques (deterministic = its best guess, no exploring)
            action, _ = model.predict(observation, deterministic=True)

            # Returns: next state, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Slow the loop slightly so it looks close to real time
            time.sleep(env.unwrapped.dt)

            # If the episode finishes (flip) or times out (1000 steps), reset it
            if terminated or truncated:
                episodes += 1
                flips += int(info["flipped"])
                status = "FLIPPED" if info["flipped"] else "stayed upright"
                print(f"Episode {episodes}: {status}, reward {episode_reward:.1f} "
                      f"(flips so far: {flips}/{episodes})")
                observation, info = env.reset()
                episode_reward = 0.0

    except KeyboardInterrupt:
        print("\nSimulation intercepted by user.")

    finally:
        print("Closing environment.")
        env.close()


def main():
    if "--train" in sys.argv or not os.path.exists(MODEL_PATH + ".zip"):
        train()
    watch()


if __name__ == "__main__":
    main()


"""
Standard 2-legged HalfCheetah (Gymnasium HalfCheetah-v5 with xml_file)

Head: fixed geom at the front of the torso (no joint, no action)
Reward: AntiFlipReward wrapper (forward + alive - control - tilt, big penalty and episode end past 85 degrees)
Policy: SAC from Stable-Baselines3, saved to cheetah_2leg_sac.zip
Action space: Box(-1, 1, (6,), float32)
All joints are hinges, unit is torque (N m).

num: Action (min,max), name in XML file
0: Torque applied on the back thigh rotor (-1,1), bthigh
1: Torque applied on the back shin rotor (-1,1), bshin
2: Torque applied on the back foot rotor (-1,1), bfoot
3: Torque applied on the front thigh rotor (-1,1), fthigh
4: Torque applied on the front shin rotor (-1,1), fshin
5: Torque applied on the front foot rotor (-1,1), ffoot
"""