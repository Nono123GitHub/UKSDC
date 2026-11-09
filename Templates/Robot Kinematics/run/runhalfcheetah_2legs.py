import os
import re
import sys
import glob
import time
import tempfile
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

# ---------------------------------------------------------------------
# GRAVITY SETTING (used when training a new model)
# ---------------------------------------------------------------------
# Strength of gravity in m/s^2 (pulls straight down)
# Earth 9.81, Moon 1.62, Mars 3.71, Jupiter 24.79, Pluto 0.62
GRAVITY = 3.71
DEFAULT_GRAVITY = 9.81   # assumed for old model files with no gravity in the name

# ---------------------------------------------------------------------
# REWARD SETTINGS
# ---------------------------------------------------------------------
REWARD_WEIGHTS = {
    "forward": 1.0,   # bonus per unit of forward speed
    "control": 0.1,   # penalty for torque use
    "alive": 1.0,     # bonus every step it hasn't flipped
    "upright": 1.0,   # penalty per radian of tilt
}
FLIP_ANGLE = np.radians(85)   # 85 degrees; beyond this counts as flipped
FLIP_PENALTY = 100.0          # one-off penalty when it flips

# ---------------------------------------------------------------------
# TRAINING / WATCHING SETTINGS
# ---------------------------------------------------------------------
TRAIN_STEPS = 200_000          # more = better, but slower (roughly 1 to 2 hours on CPU)
# Gravity is in the filename so each gravity gets its own trained model
MODEL_BASE = "cheetah_2leg"
MODEL_PATH = f"{MODEL_BASE}_g{GRAVITY:.2f}_sac"   # e.g. cheetah_2leg_g3.71_sac.zip
WATCH_EPISODES = 10            # how many episodes to show before closing

# Standard 2-legged HalfCheetah body, written out so you can customise it with the guide above
# GRAVITY_Z is replaced with the real value in write_model_file()
CHEETAH_XML = """
<mujoco model="cheetah_2leg">
  <compiler angle="radian" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <option gravity="0 0 GRAVITY_Z" timestep="0.01"/>
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


class AntiFlipReward(gym.Wrapper):
    """Replaces the built-in reward with one that rewards running and punishes flipping."""

    def step(self, action):
        # Ignore the original reward (the "_") and build our own
        obs, _, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data
        w = REWARD_WEIGHTS

        # Torso tilt in radians, wrapped to -pi..pi (0 = level)
        tilt = (data.qpos[2] + np.pi) % (2 * np.pi) - np.pi

        reward = (
            w["forward"] * info["x_velocity"]
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


def write_model_file(gravity):
    # Put the chosen gravity into the XML (negative z = pulling down)
    xml = CHEETAH_XML.replace("GRAVITY_Z", f"{-abs(gravity)}")
    # MuJoCo needs a real file path, so save the XML to a temp file
    path = os.path.join(tempfile.gettempdir(), "cheetah_2leg.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def make_env(gravity, render_mode=None):
    # v5 is required: it accepts xml_file and sizes the action/observation spaces from the model
    env = gym.make("HalfCheetah-v5", xml_file=write_model_file(gravity), render_mode=render_mode)
    return AntiFlipReward(env)


def gravity_from_filename(path):
    # Reads the "_g3.71_" part of the filename; old files without it use Earth gravity
    match = re.search(r"_g(\d+(?:\.\d+)?)_", os.path.basename(path))
    return float(match.group(1)) if match else DEFAULT_GRAVITY


def find_models():
    # Only this body's own models, so a model for a different body can't be picked by mistake
    found = glob.glob(f"{MODEL_BASE}_g*_sac.zip") + glob.glob(f"{MODEL_BASE}_sac.zip")
    return sorted(set(found))


def choose_model():
    # --model some_file.zip picks a model directly
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            if not path.endswith(".zip"):
                path += ".zip"
            if os.path.exists(path):
                return path
            print(f"Could not find {path}")
        return None

    zips = find_models()
    if not zips:
        return None
    if len(zips) == 1:
        print(f"Using the only model found: {zips[0]}")
        return zips[0]

    # More than one model: show a menu
    print("\nSaved models:")
    for n, path in enumerate(zips, start=1):
        size_mb = os.path.getsize(path) / 1_000_000
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        print(f"  {n}: {path}  (gravity {gravity_from_filename(path)} m/s^2, "
              f"{size_mb:.1f} MB, saved {modified})")
    print("  T: train a new model instead")

    while True:
        choice = input(f"Pick a model (1-{len(zips)}) or T: ").strip().lower()
        if choice == "t":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(zips):
            return zips[int(choice) - 1]
        print("Not a valid choice, try again.")


def train():
    print(f"Training 2-legged cheetah at gravity {GRAVITY} m/s^2 "
          f"for {TRAIN_STEPS} steps (no window, this takes a while)...")
    env = make_env(GRAVITY, render_mode=None)   # no rendering = much faster training
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_PATH)
    env.close()
    print(f"Saved trained model to {MODEL_PATH}.zip")
    return MODEL_PATH + ".zip"


def watch(model_file):
    gravity = gravity_from_filename(model_file)   # run it in the gravity it was trained in
    env = make_env(gravity, render_mode="human")
    model = SAC.load(model_file, env=env)

    print(f"Model: {model_file}")
    print(f"Gravity: {gravity} m/s^2")
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
    if "--train" in sys.argv:
        model_file = train()
    else:
        model_file = choose_model()
        if model_file is None:
            model_file = train()
    watch(model_file)


if __name__ == "__main__":
    main()


"""
Standard 2-legged HalfCheetah (Gymnasium HalfCheetah-v5 with xml_file)

How to run:
  python this_file.py                    menu of saved models for this body (or trains if none exist)
  python this_file.py --model name.zip   watch a specific model
  python this_file.py --train            train a new model at GRAVITY, then watch it

Gravity: new models train at GRAVITY (currently 3.71 m/s^2, Mars).
         Watching reads gravity from the filename (_g3.71_); untagged files use 9.81.
Head: fixed geom at the front of the torso (no joint, no action)
Reward: AntiFlipReward wrapper (forward + alive - control - tilt, big penalty and episode end past 85 degrees)
Policy: SAC from Stable-Baselines3, saved as cheetah_2leg_g<gravity>_sac.zip
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