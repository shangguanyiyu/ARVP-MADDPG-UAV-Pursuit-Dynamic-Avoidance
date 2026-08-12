import numpy as np
import math

# 1 simulate 3D lidar - using spherical coordinates
def update_lasers(pos, obs_pos, r, L, num_lasers, bound):
    """
    3D laser simulation using spherical coordinate sampling.
    We sample num_lasers directions distributed over the sphere (fibonacci lattice).
    pos: (x,y,z)
    obs_pos: (x,y,z)
    r: obstacle radius
    L: max laser range
    num_lasers: number of laser beams
    bound: environment side length (cube [0, bound]^3)
    """
    pos = np.array(pos)
    obs_pos = np.array(obs_pos)
    
    distance_to_obs = np.linalg.norm(pos - obs_pos)
    isInObs = (distance_to_obs < r) \
                or pos[0] < 0 or pos[0] > bound \
                or pos[1] < 0 or pos[1] > bound \
                or pos[2] < 0 or pos[2] > bound
    
    if isInObs:
        return [0.0] * num_lasers, isInObs
    
    # Generate directions using Fibonacci lattice for quasi-uniform sphere sampling
    directions = _fibonacci_sphere(num_lasers)
    laser_lengths = [L] * num_lasers
    
    for i, dir_vec in enumerate(directions):
        intersection_dist = check_obs_intersection(pos, dir_vec, obs_pos, r, L)
        if laser_lengths[i] > intersection_dist:
            laser_lengths[i] = intersection_dist
    
    for i, dir_vec in enumerate(directions):
        wall_dist = check_wall_intersection(pos, dir_vec, bound, L)
        if laser_lengths[i] > wall_dist:
            laser_lengths[i] = wall_dist
    
    return laser_lengths, isInObs


def _fibonacci_sphere(samples):
    """Generate quasi-uniform points on unit sphere using Fibonacci lattice."""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        points.append(np.array([x, y, z]))
    return points


def check_obs_intersection(start_pos, dir_vec, obs_pos, r, max_distance):
    """
    Ray-sphere intersection in 3D.
    start_pos: ray origin (x,y,z)
    dir_vec: unit direction vector (dx,dy,dz)
    obs_pos: sphere center (x,y,z)
    r: sphere radius
    max_distance: max ray length
    Returns distance along ray to intersection, or max_distance if none.
    """
    dir_vec = np.array(dir_vec, dtype=float)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
    
    ox, oy, oz = obs_pos
    sx, sy, sz = start_pos
    
    # Ray: P = start + t * dir, t in [0, max_distance]
    # Sphere: (P - obs)^2 = r^2
    dx, dy, dz = dir_vec
    fx = sx - ox
    fy = sy - oy
    fz = sz - oz

    a = dx**2 + dy**2 + dz**2
    b = 2 * (fx * dx + fy * dy + fz * dz)
    c = (fx**2 + fy**2 + fz**2) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        if 0 <= t1 <= max_distance:
            return t1
        if 0 <= t2 <= max_distance:
            return t2

    return max_distance


def check_wall_intersection(start_pos, dir_vec, bound, L):
    """
    Ray-cube intersection in 3D. Cube is [0, bound]^3.
    Returns distance along ray to nearest wall, or L if none within range.
    """
    dir_vec = np.array(dir_vec, dtype=float)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
    sx, sy, sz = start_pos
    dx, dy, dz = dir_vec
    L_ = L
    
    # 6 walls: x=0, x=bound, y=0, y=bound, z=0, z=bound
    # x = bound
    if dx > 1e-9:
        L_ = min(L_, abs((bound - sx) / dx))
    # x = 0
    if dx < -1e-9:
        L_ = min(L_, abs(sx / -dx))
    # y = bound
    if dy > 1e-9:
        L_ = min(L_, abs((bound - sy) / dy))
    # y = 0
    if dy < -1e-9:
        L_ = min(L_, abs(sy / -dy))
    # z = bound
    if dz > 1e-9:
        L_ = min(L_, abs((bound - sz) / dz))
    # z = 0
    if dz < -1e-9:
        L_ = min(L_, abs(sz / -dz))

    return L_


def cal_triangle_S(p1, p2, p3):
    """
    Compute triangle area given 3 points (used for 2D compatibility and 3D planar triangle).
    Uses cross product: S = 0.5 * ||(p2-p1) x (p3-p1)||
    Works for both 2D (fills z=0) and 3D points.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    if len(p1) == 2:
        p1 = np.append(p1, 0.0)
        p2 = np.append(p2, 0.0)
        p3 = np.append(p3, 0.0)
    v1 = p2 - p1
    v2 = p3 - p1
    S = 0.5 * np.linalg.norm(np.cross(v1, v2))
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    return S


def cal_tetrahedron_V(p1, p2, p3, p4):
    """
    Compute signed volume of tetrahedron formed by 4 3D points.
    V = (1/6) * |det([p2-p1, p3-p1, p4-p1])|
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    vol = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
    return vol


def check_encirclement_3d(hunter_positions, target_pos, capture_radius):
    """
    Check if target is encircled by hunters in 3D.
    Strategy: target is inside the convex hull of hunters AND each hunter is within capture_radius.
    Simplified: check that for every direction from target, there's a hunter within a cone angle.
    Returns: (is_encircled: bool, avg_distance: float, all_in_range: bool)
    """
    hunters = [np.array(h) for h in hunter_positions]
    target = np.array(target_pos)
    n_hunters = len(hunters)
    
    # Check capture radius
    distances = [np.linalg.norm(h - target) for h in hunters]
    all_in_range = all(d <= capture_radius for d in distances)
    avg_distance = np.mean(distances)
    
    if n_hunters < 3:
        return False, avg_distance, all_in_range
    
    # Simplified 3D encirclement check: 
    # Compute vectors from target to hunters, check spherical coverage
    # If the target "sees" hunters in enough different directions (hemispheres), encircled.
    directions = [(h - target) / (np.linalg.norm(h - target) + 1e-9) for h in hunters]
    
    # Check if any hemisphere (defined by a plane through target) has zero hunters
    # We use the hunters' mean direction as a heuristic: 
    # If mean direction magnitude is small (< 0.5), they roughly evenly distributed.
    mean_dir = np.mean(directions, axis=0)
    mean_mag = np.linalg.norm(mean_dir)
    is_encircled = (mean_mag < 0.6) and all_in_range  # threshold tunable
    
    return is_encircled, avg_distance, all_in_range


# =============================================================================
# 3D Artificial Potential Field (APF) for UAV guidance
# =============================================================================

def apf_gradient_3d(pos, target_pos, obstacle_list, other_hunter_positions,
                    bound, k_att=1.0, k_rep=0.2, k_rep_hunter=0.15, k_wall=0.3,
                    d0_obstacle=0.35, d0_hunter=0.3, d0_wall=0.25):
    """
    Compute the 3D Artificial Potential Field gradient and total potential energy
    at position `pos` for a hunter UAV in round-up task.

    Potential components:
      1. Attractive: to target (quadratic U_att = 0.5 * k_att * d^2)
         Gradient: -k_att * (pos - target)
      2. Repulsive: to obstacles (only active when d < d0_obstacle)
         U_rep = 0.5 * k_rep * (1/d - 1/d0)^2 * (d_target)^n
      3. Repulsive: to other hunter UAVs (avoid inter-hunter collision)
      4. Repulsive: to environment cube walls [0, bound]^3 (6 faces)

    Args:
      pos: (x,y,z) current UAV position
      target_pos: (x,y,z) target UAV position (attraction goal)
      obstacle_list: list of dicts [{'position':[x,y,z], 'radius':r}, ...]
      other_hunter_positions: list of other 3D hunter positions [p1, p2, ...]
      bound: cube side length [0, bound]^3
      k_att, k_rep, k_rep_hunter, k_wall: gains for attractive/repulsive terms
      d0_obstacle, d0_hunter, d0_wall: influence radii

    Returns:
      gradient_norm: (3,) unit vector of total APF gradient direction (normalized)
      total_energy:  scalar total potential energy U(pos) (for reward shaping)
    """
    pos = np.array(pos, dtype=float)
    target = np.array(target_pos, dtype=float)

    grad = np.zeros(3, dtype=float)
    U = 0.0
    diag_L = np.sqrt(3) * bound + 1e-6

    # ---- 1. Attractive potential to target ----
    diff_att = pos - target
    d_att = np.linalg.norm(diff_att) + 1e-9
    # quadratic attractive
    grad_att = k_att * diff_att  # = +k_att*(pos-target) = -∇U_att with the usual convention
    U_att = 0.5 * k_att * d_att * d_att
    grad += grad_att
    U += U_att

    # ---- 2. Repulsive potential from spherical obstacles ----
    for obs in obstacle_list:
        obs_pos = np.array(obs['position'], dtype=float)
        r = float(obs['radius'])
        diff_rep = pos - obs_pos
        d_obs = np.linalg.norm(diff_rep) + 1e-9
        # surface distance
        d_surf = max(d_obs - r, 1e-6)
        if d_surf < d0_obstacle and d_surf > 1e-6:
            # U_rep = 0.5 * k_rep * (1/d - 1/d0)^2 * scale
            inv_d = 1.0 / d_surf
            inv_d0 = 1.0 / d0_obstacle
            term = inv_d - inv_d0
            U_rep = 0.5 * k_rep * term * term
            # derivative w.r.t. d_surf: k_rep * term * (-1/d_surf^2)
            # gradient w.r.t. pos: derivative * (diff_rep / d_surf)
            dU_dd = k_rep * term * (-1.0 / (d_surf * d_surf))
            grad_rep = dU_dd * (diff_rep / (d_obs + 1e-9))  # direction away from obstacle center
            grad += grad_rep
            U += U_rep

    # ---- 3. Repulsive potential from other hunters ----
    for other_p in other_hunter_positions:
        other_p = np.array(other_p, dtype=float)
        diff_h = pos - other_p
        d_h = np.linalg.norm(diff_h) + 1e-9
        if d_h < d0_hunter:
            inv_d = 1.0 / d_h
            inv_d0 = 1.0 / d0_hunter
            term = inv_d - inv_d0
            U_rep_h = 0.5 * k_rep_hunter * term * term
            dU_dd = k_rep_hunter * term * (-1.0 / (d_h * d_h))
            grad_rep_h = dU_dd * (diff_h / (d_h + 1e-9))
            grad += grad_rep_h
            U += U_rep_h

    # ---- 4. Repulsive potential from 6 cube walls (x=0, x=bound, y=0, y=bound, z=0, z=bound) ----
    walls = [
        # (distance_to_wall, outward_unit_normal)
        (pos[0],                np.array([-1., 0., 0.])),   # x=0 normal points -x
        (bound - pos[0],        np.array([ 1., 0., 0.])),   # x=L normal points +x
        (pos[1],                np.array([ 0.,-1., 0.])),   # y=0 normal points -y
        (bound - pos[1],        np.array([ 0., 1., 0.])),   # y=L normal points +y
        (pos[2],                np.array([ 0., 0.,-1.])),   # z=0 normal points -z
        (bound - pos[2],        np.array([ 0., 0., 1.])),   # z=L normal points +z
    ]
    for d_w, normal in walls:
        d_w = max(d_w, 1e-6)
        if d_w < d0_wall:
            inv_d = 1.0 / d_w
            inv_d0 = 1.0 / d0_wall
            term = inv_d - inv_d0
            U_w = 0.5 * k_wall * term * term
            # dU/dd = k_wall * term * (-1/d^2). Gradient points along normal (away from wall).
            dU_dd = k_wall * term * (-1.0 / (d_w * d_w))
            grad_w = dU_dd * normal  # normal already points outward
            grad += grad_w
            U += U_w

    # Normalize gradient to unit direction for observation stability
    gnorm = np.linalg.norm(grad)
    if gnorm > 1e-9:
        gradient_norm = grad / gnorm
    else:
        gradient_norm = np.zeros(3)

    return gradient_norm, U


def apf_evasive_gradient_3d(pos, hunter_positions, bound,
                            k_rep_hunter=1.5, k_att_center=0.2, k_wall=0.3,
                            d0_hunter=0.6, d0_wall=0.3):
    """
    Compute 3D APF gradient for the TARGET (evasive) UAV:
      - Repulsive: away from all hunters (encourage escaping)
      - Weak attractive: toward cube center (prevent hugging walls)
      - Repulsive: from walls
    Returns (gradient_norm_3d, total_energy_U).
    """
    pos = np.array(pos, dtype=float)
    center = np.array([bound/2., bound/2., bound/2.])

    grad = np.zeros(3, dtype=float)
    U = 0.0

    # Repel from hunters
    for h_p in hunter_positions:
        h_p = np.array(h_p, dtype=float)
        diff_h = pos - h_p
        d_h = np.linalg.norm(diff_h) + 1e-9
        if d_h < d0_hunter:
            inv_d = 1.0 / d_h
            inv_d0 = 1.0 / d0_hunter
            term = inv_d - inv_d0
            U_rep_h = 0.5 * k_rep_hunter * term * term
            dU_dd = k_rep_hunter * term * (-1.0 / (d_h * d_h))
            grad_rep_h = dU_dd * (diff_h / (d_h + 1e-9))
            grad += grad_rep_h
            U += U_rep_h

    # Weak attractive to center
    diff_c = pos - center
    d_c = np.linalg.norm(diff_c) + 1e-9
    grad += k_att_center * diff_c
    U += 0.5 * k_att_center * d_c * d_c

    # Wall repulsion (same as hunter, 6 faces)
    walls = [
        (pos[0],         np.array([-1., 0., 0.])),
        (bound - pos[0], np.array([ 1., 0., 0.])),
        (pos[1],         np.array([ 0.,-1., 0.])),
        (bound - pos[1], np.array([ 0., 1., 0.])),
        (pos[2],         np.array([ 0., 0.,-1.])),
        (bound - pos[2], np.array([ 0., 0., 1.])),
    ]
    for d_w, normal in walls:
        d_w = max(d_w, 1e-6)
        if d_w < d0_wall:
            inv_d = 1.0 / d_w
            inv_d0 = 1.0 / d0_wall
            term = inv_d - inv_d0
            U_w = 0.5 * k_wall * term * term
            dU_dd = k_wall * term * (-1.0 / (d_w * d_w))
            grad_w = dU_dd * normal
            grad += grad_w
            U += U_w

    gnorm = np.linalg.norm(grad)
    gradient_norm = (grad / gnorm) if gnorm > 1e-9 else np.zeros(3)
    return gradient_norm, U
