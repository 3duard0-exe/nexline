import os
import sys
import math
import random
import argparse
import time
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# =========================
# Utility and Config
# =========================
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def angle_diff(a, b):
    # Smallest signed angle between a and b
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))

# =========================
# Track Generator (no self-crossing, curriculum-ready)
# =========================
def _build_monotonic_points(width, height, margin, n_pts, roughness, acute=False):
    """
    Build a polyline with strictly increasing x, and y from a bounded random walk.
    This guarantees no self-intersections.
    - roughness controls typical step in y.
    - acute=True injects occasional big y jumps.
    """
    xs = np.linspace(margin, width - margin, n_pts)
    y = random.randint(margin, height - margin)
    ys = [y]
    for i in range(1, n_pts):
        dy = np.random.normal(0, roughness)
        if acute and random.random() < 0.15:
            dy += random.choice([-1, 1]) * random.uniform(3*roughness, 5*roughness)
        y = clamp(ys[-1] + dy, margin, height - margin)
        ys.append(y)
    points = [(float(xs[i]), float(ys[i])) for i in range(n_pts)]
    return points

def catmull_rom_spline(points, samples_per_segment=20):
    """Generate a smooth curve through points using Catmull-Rom spline (x stays monotonic if inputs are monotonic)."""
    if len(points) < 4:
        return points
    curve = []
    pts = [points[0]] + points + [points[-1]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        for t_i in range(samples_per_segment):
            t = t_i / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2*p1[0]) +
                       (-p0[0] + p2[0]) * t +
                       (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2*p1[1]) +
                       (-p0[1] + p2[1]) * t +
                       (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            curve.append((x, y))
    return curve

def generate_track(width, height, level='linear', line_width=None):
    """
    Auto-generate a track according to curriculum level without self-crossing:
    - 'linear': mostly straight, small y variations, piecewise linear.
    - 'smooth': smooth curvy line via Catmull-Rom on monotonic points.
    - 'acute': abrupt y changes but still monotonic x (no crossing).
    """
    if line_width is None:
        line_width = random.randint(4, 8)  # thinner line

    bg_color = (255, 255, 255)
    line_color = (0, 0, 0)
    margin = 40

    surf = pygame.Surface((width, height))
    surf.fill(bg_color)

    if level == 'linear':
        n_pts = random.randint(20, 35)
        pts = _build_monotonic_points(width, height, margin, n_pts, roughness=8.0, acute=False)
        centerline = pts

    elif level == 'smooth':
        n_pts = random.randint(8, 12)
        ctrl = _build_monotonic_points(width, height, margin, n_pts, roughness=20.0, acute=False)
        centerline = catmull_rom_spline(ctrl, samples_per_segment=25)

    elif level == 'acute':
        n_pts = random.randint(25, 45)
        pts = _build_monotonic_points(width, height, margin, n_pts, roughness=14.0, acute=True)
        centerline = pts

    else:
        n_pts = random.randint(8, 12)
        ctrl = _build_monotonic_points(width, height, margin, n_pts, roughness=20.0, acute=False)
        centerline = catmull_rom_spline(ctrl, samples_per_segment=25)

    if len(centerline) < 2:
        centerline = [(margin, height//2), (width - margin, height//2)]

    pygame.draw.lines(surf, line_color, False, [(int(x), int(y)) for (x, y) in centerline], line_width)
    return surf, centerline, line_width

# =========================
# Line Follower Environment
# =========================
class LineFollowerEnv:
    def __init__(self, width=800, height=600, render=True, fps=60, seed=None,
                 speed_scale=90.0, turn_rate_factor=0.25, turn_boost=1.0,
                 sensor_spread_factor=2.2, sensor_forward_factor=0.8,
                 accel_rate=3.0, min_forward_base=0.12,
                 # Adaptive control coefficients
                 adapt_speed_err=0.5, adapt_speed_heading=0.6, adapt_speed_online_bonus=0.25,
                 speed_factor_min=0.35, speed_factor_max=1.15,
                 adapt_turn_err=0.9, adapt_turn_heading=1.2, adapt_turn_offline=0.8,
                 turn_factor_min=0.8, turn_factor_max=2.4,
                 lookahead_k=4):
        if not render and os.environ.get("SDL_VIDEODRIVER") is None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.width = width
        self.height = height
        self.render_enabled = bool(render)
        self.fps = fps
        self.dt = 1.0 / fps
        self.rng = random.Random(seed if seed is not None else time.time())
        self.bg_color = (255, 255, 255)
        self.line_color = (0, 0, 0)
        self.screen = None
        self.clock = None

        # Car parameters
        self.car_len = 30
        self.car_wid = 20
        self.wheel_base = 22.0
        self.speed_scale = float(speed_scale)
        self.max_speed = 1.0
        self.min_speed = -0.5
        # Turning rate and boost
        self.turn_rate_factor = float(turn_rate_factor)
        self.turn_boost = float(turn_boost)

        # Acceleration and anti-stall
        self.accel_rate = float(accel_rate)
        self.min_forward_base = float(min_forward_base)

        # Sensors
        self.sensor_spread_factor = float(sensor_spread_factor)
        self.sensor_forward_factor = float(sensor_forward_factor)
        self.sensor_positions_local = self._build_sensor_array(
            spread_factor=self.sensor_spread_factor, forward_factor=self.sensor_forward_factor
        )
        self.sensor_threshold = 120

        # Adaptive control
        self.adapt_speed_err = float(adapt_speed_err)
        self.adapt_speed_heading = float(adapt_speed_heading)
        self.adapt_speed_online_bonus = float(adapt_speed_online_bonus)
        self.speed_factor_min = float(speed_factor_min)
        self.speed_factor_max = float(speed_factor_max)

        self.adapt_turn_err = float(adapt_turn_err)
        self.adapt_turn_heading = float(adapt_turn_heading)
        self.adapt_turn_offline = float(adapt_turn_offline)
        self.turn_factor_min = float(turn_factor_min)
        self.turn_factor_max = float(turn_factor_max)
        self.lookahead_k = int(lookahead_k)

        # Episode and track
        self.map_surf = None
        self.centerline = None
        self.line_width = None

        # Car state
        self.x = self.width // 2
        self.y = self.height // 2
        self.theta = 0.0
        self.vl = 0.0
        self.vr = 0.0
        self.dead_counter = 0
        self.dead_limit = 24

        # Progress tracking
        self.progress_best_idx = 0
        self.finish_radius = 14.0
        self.curriculum_level = 'linear'

        # Stall tracking to prevent "parking"
        self.stall_counter = 0
        self.stall_limit = int(4.0 * self.fps)  # 4 seconds without progress

        pygame.init()
        if self.render_enabled:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Line Follower DQN (8 sensors)")
            self.clock = pygame.time.Clock()

    def set_curriculum(self, level):
        self.curriculum_level = level

    def _build_sensor_array(self, spread_factor=1.6, forward_factor=0.6):
        front_x = self.car_len * forward_factor
        spread = self.car_wid * spread_factor
        ys = np.linspace(-spread/2, spread/2, 8)
        return [(front_x, float(y)) for y in ys]

    def _rotate(self, x_local, y_local, theta):
        c, s = math.cos(theta), math.sin(theta)
        return (x_local * c - y_local * s, x_local * s + y_local * c)

    def _sensor_world_positions(self):
        positions = []
        for (sx, sy) in self.sensor_positions_local:
            rx, ry = self._rotate(sx, sy, self.theta)
            positions.append((self.x + rx, self.y + ry))
        return positions

    def _read_sensors(self):
        positions = self._sensor_world_positions()
        readings = []
        for (px, py) in positions:
            ix, iy = int(px), int(py)
            if not (0 <= ix < self.width and 0 <= iy < self.height):
                readings.append(0.0)
                continue
            color = self.map_surf.get_at((ix, iy))
            gray = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
            val = 1.0 if gray < self.sensor_threshold else 0.0
            readings.append(val)
        return np.array(readings, dtype=np.float32)

    def _nearest_centerline_idx(self, step_stride=1):
        min_dist_sq = 1e9
        min_idx = 0
        for i in range(0, len(self.centerline), step_stride):
            cx, cy = self.centerline[i]
            d_sq = (self.x - cx)**2 + (self.y - cy)**2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                min_idx = i
        return min_idx, math.sqrt(min_dist_sq)

    def _desired_heading_at(self, idx):
        j = min(idx + self.lookahead_k, len(self.centerline) - 1)
        i = max(0, idx)
        if j == i:  # degenerate
            return self.theta
        p1 = self.centerline[i]
        p2 = self.centerline[j]
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    def reset(self):
        self.map_surf, self.centerline, self.line_width = generate_track(
            self.width, self.height, level=self.curriculum_level)
        p0 = self.centerline[0]
        p1 = self.centerline[min(1, len(self.centerline)-1)]
        dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
        self.x, self.y = p0
        self.theta = math.atan2(dy, dx)
        self.vl = 0.0
        self.vr = 0.0
        self.dead_counter = 0
        self.progress_best_idx = 0
        self.stall_counter = 0
        return self._read_sensors()

    def step(self, action_idx):
        # Corrected action map:
        # 0: slow fwd, 1: medium fwd, 2: fast fwd,
        # 3: right medium (omega negative), 4: right hard pivot (omega negative),
        # 5: left medium (omega positive), 6: left hard pivot (omega positive)
        action_map = [
            (0.25, 0.25), (0.6, 0.6), (1.0, 1.0),
            (0.9, 0.4), (1.0, -0.2),
            (0.4, 0.9), (-0.2, 1.0),
        ]
        action_idx = int(clamp(action_idx, 0, len(action_map)-1))
        tgt_vl, tgt_vr = action_map[action_idx]

        # Accelerate wheel speeds towards targets
        tgt_vl = clamp(tgt_vl, self.min_speed, self.max_speed)
        tgt_vr = clamp(tgt_vr, self.min_speed, self.max_speed)
        max_delta = self.accel_rate * self.dt
        self.vl += clamp(tgt_vl - self.vl, -max_delta, max_delta)
        self.vr += clamp(tgt_vr - self.vr, -max_delta, max_delta)

        # Compute basic forward magnitude and anti-stall baseline
        forward_mag = (self.vl + self.vr) * 0.5
        if forward_mag < self.min_forward_base:
            delta = self.min_forward_base - forward_mag
            self.vl = clamp(self.vl + delta, self.min_speed, self.max_speed)
            self.vr = clamp(self.vr + delta, self.min_speed, self.max_speed)
            forward_mag = (self.vl + self.vr) * 0.5

        # Read sensors & nearest centerline to build adaptive gains BEFORE moving
        obs_pre = self._read_sensors()
        sum_s_pre = float(np.sum(obs_pre))
        line_term_pre = sum_s_pre / 8.0
        abs_err = 0.0
        if sum_s_pre > 0.0:
            idxs = np.arange(8, dtype=np.float32)
            c = float(np.sum(idxs * obs_pre) / sum_s_pre)
            abs_err = abs((c - 3.5) / 3.5)  # ~[0,1]

        near_idx_pre, _ = self._nearest_centerline_idx(step_stride=2)
        desired_theta = self._desired_heading_at(near_idx_pre)
        heading_mis = abs(angle_diff(desired_theta, self.theta)) / math.pi  # normalize to [0,1]

        # Adaptive speed and turn factors
        speed_factor = (
            1.0
            - self.adapt_speed_err * abs_err
            - self.adapt_speed_heading * heading_mis
            + self.adapt_speed_online_bonus * line_term_pre
        )
        speed_factor = clamp(speed_factor, self.speed_factor_min, self.speed_factor_max)

        turn_factor = (
            1.0
            + self.adapt_turn_err * abs_err
            + self.adapt_turn_heading * heading_mis
            + self.adapt_turn_offline * (1.0 - line_term_pre)
        )
        turn_factor = clamp(turn_factor, self.turn_factor_min, self.turn_factor_max)

        # Kinematics: adaptive speed and rotation
        v = forward_mag * self.speed_scale * speed_factor
        base_omega = (self.vr - self.vl) / self.wheel_base * self.speed_scale * self.turn_rate_factor
        diff = min(1.0, abs(self.vr - self.vl))
        omega = base_omega * (1.0 + self.turn_boost * diff) * turn_factor

        # Integrate motion
        self.theta += omega * self.dt
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.x = clamp(self.x, 1, self.width - 2)
        self.y = clamp(self.y, 1, self.height - 2)

        # Observation AFTER moving for agent
        obs = self._read_sensors()
        sum_s = float(np.sum(obs))
        reward = -0.01

        if sum_s > 0:
            idxs = np.arange(8, dtype=np.float32)
            c = float(np.sum(idxs * obs) / sum_s)
            center_term = 1.0 - min(abs(c - 3.5) / 3.5, 1.0)
            line_term = sum_s / 8.0
            reward += 0.6 * center_term + 0.4 * line_term
            self.dead_counter = 0
        else:
            reward -= 0.05
            self.dead_counter += 1

        # Encourage movement; discourage crawling
        reward += 0.03 * max(0.0, forward_mag)
        if forward_mag < 0.12:
            reward -= 0.06

        # Reward heading alignment with desired centerline tangent
        heading_mis_post = abs(angle_diff(self._desired_heading_at(self._nearest_centerline_idx(2)[0]), self.theta)) / math.pi
        reward += 0.02 * (1.0 - heading_mis_post)

        # Soft penalty for differential (don’t suppress turning too much)
        reward -= 0.004 * abs(self.vr - self.vl)

        # Progress tracking using POST-move position
        near_idx, near_dist = self._nearest_centerline_idx(step_stride=2)
        if near_idx > self.progress_best_idx:
            progress_gain = near_idx - self.progress_best_idx
            reward += 0.08 * progress_gain
            self.progress_best_idx = near_idx
            self.stall_counter = 0
        else:
            self.stall_counter += 1

        done = False
        if near_idx >= len(self.centerline) - 2 and near_dist < self.finish_radius:
            reward += 8.0
            done = True
        if self.dead_counter >= self.dead_limit:
            reward -= 1.0
            done = True
        if self.stall_counter >= self.stall_limit:
            reward -= 0.5
            done = True
        if (int(self.x) <= 1 or int(self.x) >= self.width - 2 or \
            int(self.y) <= 1 or int(self.y) >= self.height - 2) and sum_s == 0:
            reward -= 0.5
            done = True

        return obs, reward, done, {}

    def render(self):
        if not self.render_enabled:
            return True  # Continue running (headless)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to stop running

        self.screen.blit(self.map_surf, (0, 0))

        car_color = (0, 128, 255)
        c, s = math.cos(self.theta), math.sin(self.theta)
        half_len, half_wid = self.car_len / 2, self.car_wid / 2
        pts_local = [
            (+half_len, +half_wid), (+half_len, -half_wid),
            (-half_len, -half_wid), (-half_len, +half_wid)
        ]
        pts_world = []
        for (lx, ly) in pts_local:
            rx, ry = lx * c - ly * s, lx * s + ly * c
            pts_world.append((self.x + rx, self.y + ry))
        pygame.draw.polygon(self.screen, car_color, pts_world, width=0)

        sensor_color_on, sensor_color_off = (255, 0, 0), (0, 180, 0)
        readings = self._read_sensors()
        positions = self._sensor_world_positions()
        for i, (px, py) in enumerate(positions):
            col = sensor_color_on if readings[i] > 0.5 else sensor_color_off
            pygame.draw.circle(self.screen, col, (int(px), int(py)), 3)

        fx, fy = self.centerline[-1]
        pygame.draw.circle(self.screen, (0, 200, 0), (int(fx), int(fy)), max(3, int(self.finish_radius / 2)), width=2)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True  # Continue running

    def close(self):
        if self.render_enabled:
            pygame.quit()

# =========================
# DQN Agent
# =========================
class DQN(nn.Module):
    def __init__(self, n_inputs=8, n_actions=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        return Transition(*zip(*random.sample(self.buffer, batch_size)))
    def __len__(self):
        return len(self.buffer)

def resolve_device(pref='auto'):
    if pref == 'mps' and torch.backends.mps.is_available(): return 'mps'
    if pref == 'cuda' and torch.cuda.is_available(): return 'cuda'
    if pref == 'cpu': return 'cpu'
    if torch.backends.mps.is_available(): return 'mps'
    if torch.cuda.is_available(): return 'cuda'
    return 'cpu'

class DQNAgent:
    def __init__(self, n_inputs=8, n_actions=7, lr=1e-3, gamma=0.99, device=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device or resolve_device('auto')
        self.qnet = DQN(n_inputs, n_actions).to(self.device)
        self.tgt = DQN(n_inputs, n_actions).to(self.device)
        self.tgt.load_state_dict(self.qnet.state_dict())
        self.optim = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(capacity=60000)
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay_steps = 50000
        self.learn_step = 0
        self.tgt_update_period = 2000
        self.batch_size = 64
        self.prefer_forward_on_ties = True
        print(f"Using device: {self.device}")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.qnet(s).squeeze(0)
            if self.prefer_forward_on_ties:
                q_range = (q.max() - q.min()).item()
                if not np.isfinite(q_range) or q_range < 1e-6:
                    return 1  # prefer small forward on ties
            return int(torch.argmax(q).item())

    def push_transition(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.qnet(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.qnet(next_states)
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            next_q_tgt = self.tgt(next_states).gather(1, next_actions)
            target = rewards + (1.0 - dones) * self.gamma * next_q_tgt

        loss = self.loss_fn(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.optim.step()

        self.learn_step += 1
        self.epsilon = max(self.eps_min, 1.0 - self.learn_step / float(self.eps_decay_steps))
        if self.learn_step % self.tgt_update_period == 0:
            self.tgt.load_state_dict(self.qnet.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.qnet.state_dict(), path)
    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(sd)
        self.tgt.load_state_dict(sd)

    def export_onnx(self, path='dqn_line_follower.onnx'):
        cpu_model = DQN(n_inputs=8, n_actions=self.n_actions).to('cpu')
        cpu_model.load_state_dict(self.qnet.state_dict())
        cpu_model.eval()
        dummy = torch.randn(1, 8)
        torch.onnx.export(
            cpu_model, dummy, path,
            input_names=['sensors'], output_names=['q_values'],
            dynamic_axes={'sensors': {0: 'batch'}, 'q_values': {0: 'batch'}},
            opset_version=11
        )
        print(f"Exported ONNX to {path}")

# =========================
# Simple Autopilot (for visualization without training)
# =========================
def autopilot_action(obs):
    sum_s = float(np.sum(obs))
    if sum_s <= 0.0: return 1  # small forward
    c = float(np.sum(np.arange(8, dtype=np.float32) * obs) / sum_s)
    error = (c - 3.5) / 3.5
    # Corrected mapping: positive error -> right turn (omega negative), negative error -> left turn (omega positive)
    if error > 0.6: return 4      # hard right pivot
    elif error > 0.25: return 3   # medium right
    elif error < -0.6: return 6   # hard left pivot
    elif error < -0.25: return 5  # medium left
    else: return 2                 # fast forward

def run_viewer(env, max_steps=5000):
    obs = env.reset()
    running = True
    for _ in range(max_steps):
        if not running: break
        action = autopilot_action(obs)
        obs, _, done, _ = env.step(action)
        running = env.render()
        if done:
            obs = env.reset()

# =========================
# Training / Demo Loops with Curriculum
# =========================
def curriculum_level_for_episode(ep, total_episodes):
    if ep <= 0.4 * total_episodes: return 'linear'
    elif ep <= 0.75 * total_episodes: return 'smooth'
    else: return 'acute'

def train(env, agent, episodes=300, max_steps=1500, render=False, save_path='dqn_line_follower.pth', log_every=10):
    rewards_hist = []
    running = True
    for ep in range(1, episodes + 1):
        if not running: break
        level = curriculum_level_for_episode(ep, episodes)
        env.set_curriculum(level)
        obs = env.reset()
        ep_reward, losses = 0.0, []
        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            if render:
                running = env.render()
                if not running: break
            agent.push_transition(obs, action, reward, next_obs, done)
            loss_val = agent.update()
            if loss_val is not None: losses.append(loss_val)
            ep_reward += reward
            obs = next_obs
            if done: break
        rewards_hist.append(ep_reward)
        if ep % log_every == 0:
            avg_r = np.mean(rewards_hist[-log_every:])
            avg_l = np.mean(losses) if losses else 0.0
            print(f"Episode {ep}/{episodes} | level={level} | avg_reward={avg_r:.3f} | loss={avg_l:.4f} | epsilon={agent.epsilon:.3f}")
    agent.save(save_path)
    print(f"Saved model to {save_path}")
    return rewards_hist

def demo(env, agent, episodes=5, max_steps=2000, render=True, level='smooth'):
    agent.epsilon = 0.0
    env.set_curriculum(level)
    running = True
    for ep in range(1, episodes + 1):
        if not running: break
        obs = env.reset()
        total_r = 0.0
        for _ in range(max_steps):
            action = agent.select_action(obs)
            if action == 0:  # ensure minimum forward motion
                action = 1
            obs, r, done, _ = env.step(action)
            if render:
                running = env.render()
                if not running: break
            total_r += r
            if done: break
        if running:
            print(f"Demo Episode {ep} (level={level}): reward={total_r:.3f}")

def demo_levels(env, agent, episodes_per_level=2, max_steps=1500, render=True):
    agent.epsilon = 0.0
    levels = ['linear', 'smooth', 'acute']
    running = True
    for level in levels:
        if not running: break
        env.set_curriculum(level)
        for ep in range(1, episodes_per_level + 1):
            if not running: break
            obs = env.reset()
            total_r = 0.0
            for _ in range(max_steps):
                action = agent.select_action(obs)
                if action == 0:
                    action = 1
                obs, r, done, _ = env.step(action)
                if render:
                    running = env.render()
                    if not running: break
                total_r += r
                if done: break
            if running:
                print(f"[{level}] Demo Episode {ep}: reward={total_r:.3f}")

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--train_episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=1500)
    parser.add_argument('--model_path', type=str, default='dqn_line_follower.pth')
    parser.add_argument('--export', type=int, default=1)
    parser.add_argument('--demo', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--demo_level', type=str, default='smooth', choices=['linear', 'smooth', 'acute', 'all'])
    parser.add_argument('--episodes_per_level', type=int, default=2)
    # Tunables for rotation/sensors/acceleration
    parser.add_argument('--speed_scale', type=float, default=90.0)
    parser.add_argument('--turn_rate', type=float, default=0.25)
    parser.add_argument('--turn_boost', type=float, default=1.0)
    parser.add_argument('--sensor_spread', type=float, default=2.2)
    parser.add_argument('--sensor_forward', type=float, default=0.8)
    parser.add_argument('--accel_rate', type=float, default=3.0)
    parser.add_argument('--min_forward', type=float, default=0.12)
    # Adaptive coefficients
    parser.add_argument('--adapt_speed_err', type=float, default=0.5)
    parser.add_argument('--adapt_speed_heading', type=float, default=0.6)
    parser.add_argument('--adapt_speed_bonus', type=float, default=0.25)
    parser.add_argument('--speed_factor_min', type=float, default=0.35)
    parser.add_argument('--speed_factor_max', type=float, default=1.15)
    parser.add_argument('--adapt_turn_err', type=float, default=0.9)
    parser.add_argument('--adapt_turn_heading', type=float, default=1.2)
    parser.add_argument('--adapt_turn_offline', type=float, default=0.8)
    parser.add_argument('--turn_factor_min', type=float, default=0.8)
    parser.add_argument('--turn_factor_max', type=float, default=2.4)
    parser.add_argument('--lookahead_k', type=int, default=4)
    args = parser.parse_args()

    env = LineFollowerEnv(
        width=args.width, height=args.height, render=bool(args.render), fps=args.fps, seed=args.seed,
        speed_scale=args.speed_scale, turn_rate_factor=args.turn_rate, turn_boost=args.turn_boost,
        sensor_spread_factor=args.sensor_spread, sensor_forward_factor=args.sensor_forward,
        accel_rate=args.accel_rate, min_forward_base=args.min_forward,
        adapt_speed_err=args.adapt_speed_err, adapt_speed_heading=args.adapt_speed_heading,
        adapt_speed_online_bonus=args.adapt_speed_bonus, speed_factor_min=args.speed_factor_min,
        speed_factor_max=args.speed_factor_max, adapt_turn_err=args.adapt_turn_err,
        adapt_turn_heading=args.adapt_turn_heading, adapt_turn_offline=args.adapt_turn_offline,
        turn_factor_min=args.turn_factor_min, turn_factor_max=args.turn_factor_max,
        lookahead_k=args.lookahead_k
    )
    device = resolve_device(args.device)
    agent = DQNAgent(device=device)

    # Viewer mode: simple autopilot
    if bool(args.render) and args.train_episodes == 0 and args.demo == 0:
        env.set_curriculum('smooth')
        print("Viewer mode: rendering with simple autopilot.")
        run_viewer(env, max_steps=args.max_steps if args.max_steps > 0 else 5000)

    if args.train_episodes > 0:
        print("Starting training...")
        train(env, agent, episodes=args.train_episodes, max_steps=args.max_steps, render=bool(args.render), save_path=args.model_path)

    if args.export and args.train_episodes > 0:
        agent.export_onnx()
        print("To deploy on ESP32, you can convert ONNX -> TFLite -> TFLite Micro.")

    if args.demo:
        if os.path.exists(args.model_path):
            agent.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
        else:
            print(f"Model file '{args.model_path}' not found; running with untrained agent.")
        if args.demo_level == 'all':
            demo_levels(env, agent, episodes_per_level=args.episodes_per_level, max_steps=args.max_steps, render=bool(args.render))
        else:
            demo(env, agent, episodes=5, max_steps=args.max_steps, render=bool(args.render), level=args.demo_level)

    env.close()

if __name__ == '__main__':
    main()
