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
MODEL_PATH = "cheetah_3leg_midhead_jump_sac"   # saved as cheetah_3leg_midhead_jump_sac.zip
WATCH_EPISODES = 10            # how many episodes to show before closing

# Custom HalfCheetah model: the standard cheetah plus a third "middle" leg,
# with the head moved to the middle of the torso
CHEETAH_XML = """
<mujoco model="cheetah_3leg_midhead">
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
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="1500 60" texture="texplane"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="1000 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>

      <!-- Head: moved to the middle of the torso (x = 0) -->
      <geom axisangle="0 1 0 .87" name="head" pos="0 0 .1" size="0.046 .15" type="capsule"/>

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

      <!-- Middle leg (copy of the front leg, mounted at the centre of the torso) -->
      <body name="mthigh" pos="0 0.1 0">
        <joint axis="0 1 0" damping="4.5" name="mthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="mthigh" pos="-.07 0 -.12" rgba="0.4 0.6 0.9 1" size="0.046 .133" type="capsule"/>
        <body name="mshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="mshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="mshin" pos=".065 0 -.09" rgba="0.4 0.6 0.9 1" size="0.046 .106" type="capsule"/>
          <body name="mfoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="mfoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="mfoot" pos=".045 0 -.07" rgba="0.4 0.6 0.9 1" size="0.046 .07" type="capsule"/>
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
    <motor gear="120" joint="mthigh" name="mthigh"/>
    <motor gear="60"  joint="mshin"  name="mshin"/>
    <motor gear="30"  joint="mfoot"  name="mfoot"/>
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
    path = os.path.join(tempfile.gettempdir(), "cheetah_3leg_midhead.xml")
    with open(path, "w") as f:
        f.write(CHEETAH_XML)
    return path


def make_env(render_mode=None):
    # v5 is required: it accepts xml_file and sizes the action/observation spaces from the model
    env = gym.make("HalfCheetah-v5", xml_file=write_model_file(), render_mode=render_mode)
    return JumpReward(env)


def train():
    print(f"Training 3-legged middle-head cheetah for {TRAIN_STEPS} steps (no window, this takes a while)...")
    env = make_env(render_mode=None)   # no rendering = much faster training
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    env.close()
    print(f"Saved trained model to {MODEL_PATH}.zip")


def watch():
    env = make_env(render_mode="human")
    model = SAC.load(MODEL_PATH, env=env)

    print(f"Action space: {env.action_space}")            # Box(-1, 1, (9,), float32)
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
Custom 3-legged HalfCheetah with middle head (Gymnasium HalfCheetah-v5 with xml_file)

Head: fixed geom at the middle of the torso (no joint, no action)
Reward: AntiFlipReward wrapper (forward + alive - control - tilt, big penalty and episode end past 85 degrees)
Policy: SAC from Stable-Baselines3, saved to cheetah_3leg_midhead_sac.zip
Action space: Box(-1, 1, (9,), float32)
All joints are hinges, unit is torque (N m).

num: Action (min,max), name in XML file
0: Torque applied on the back thigh rotor (-1,1), bthigh
1: Torque applied on the back shin rotor (-1,1), bshin
2: Torque applied on the back foot rotor (-1,1), bfoot
3: Torque applied on the front thigh rotor (-1,1), fthigh
4: Torque applied on the front shin rotor (-1,1), fshin
5: Torque applied on the front foot rotor (-1,1), ffoot
6: Torque applied on the middle thigh rotor (-1,1), mthigh
7: Torque applied on the middle shin rotor (-1,1), mshin
8: Torque applied on the middle foot rotor (-1,1), mfoot
"""