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
