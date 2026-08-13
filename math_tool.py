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
    Compute the 3D Artificial Potential Field for a hunter UAV.

    Returns a dict with separate components for 9-dim observation augmentation:
      f_att_dir:   (3,) unit attraction force direction (toward target)
      f_rep_dir:   (3,) unit total repulsion force direction (obstacles+hunters+walls)
      U_total:     scalar total potential energy (for reward shaping)
      U_att:       scalar attraction potential energy
      U_rep:       scalar repulsion potential energy
      grad_total:  (3,) total gradient (att + rep), for alignment reward
    """
    pos = np.array(pos, dtype=float)
    target = np.array(target_pos, dtype=float)

    # ---- 1. Attractive potential to target ----
    diff_att = pos - target
    d_att = np.linalg.norm(diff_att) + 1e-9
    f_att = -k_att * diff_att  # force = -gradient = toward target
    U_att = 0.5 * k_att * d_att * d_att

    # ---- 2. Repulsive potential from spherical obstacles ----
    f_rep = np.zeros(3, dtype=float)
    U_rep = 0.0
    for obs in obstacle_list:
        obs_pos = np.array(obs['position'], dtype=float)
        r = float(obs['radius'])
        diff_rep = pos - obs_pos
        d_obs = np.linalg.norm(diff_rep) + 1e-9
        d_surf = max(d_obs - r, 1e-6)
        if d_surf < d0_obstacle and d_surf > 1e-6:
            inv_d = 1.0 / d_surf
            inv_d0 = 1.0 / d0_obstacle
            term = inv_d - inv_d0
            U_rep += 0.5 * k_rep * term * term
            # force = -gradient = away from obstacle
            dU_dd = k_rep * term * (-1.0 / (d_surf * d_surf))
            f_rep += -dU_dd * (diff_rep / (d_obs + 1e-9))

    # ---- 3. Repulsive potential from other hunters ----
    for other_p in other_hunter_positions:
        other_p = np.array(other_p, dtype=float)
        diff_h = pos - other_p
        d_h = np.linalg.norm(diff_h) + 1e-9
        if d_h < d0_hunter:
            inv_d = 1.0 / d_h
            inv_d0 = 1.0 / d0_hunter
            term = inv_d - inv_d0
            U_rep += 0.5 * k_rep_hunter * term * term
            dU_dd = k_rep_hunter * term * (-1.0 / (d_h * d_h))
            f_rep += -dU_dd * (diff_h / (d_h + 1e-9))

    # ---- 4. Repulsive potential from 6 cube walls ----
    walls = [
        (pos[0],                np.array([-1., 0., 0.])),
        (bound - pos[0],        np.array([ 1., 0., 0.])),
        (pos[1],                np.array([ 0.,-1., 0.])),
        (bound - pos[1],        np.array([ 0., 1., 0.])),
        (pos[2],                np.array([ 0., 0.,-1.])),
        (bound - pos[2],        np.array([ 0., 0., 1.])),
    ]
    for d_w, normal in walls:
        d_w = max(d_w, 1e-6)
        if d_w < d0_wall:
            inv_d = 1.0 / d_w
            inv_d0 = 1.0 / d0_wall
            term = inv_d - inv_d0
            U_rep += 0.5 * k_wall * term * term
            dU_dd = k_wall * term * (-1.0 / (d_w * d_w))
            f_rep += -dU_dd * normal

    # Normalize directions
    f_att_dir = f_att / (np.linalg.norm(f_att) + 1e-9)
    f_rep_dir = f_rep / (np.linalg.norm(f_rep) + 1e-9) if np.linalg.norm(f_rep) > 1e-9 else np.zeros(3)

    U_total = U_att + U_rep
    # Total gradient for reward shaping (gradient of U, NOT force)
    grad_total = -f_att - f_rep  # gradient = -force

    return {
        'f_att_dir': f_att_dir,
        'f_rep_dir': f_rep_dir,
        'U_total': float(U_total),
        'U_att': float(U_att),
        'U_rep': float(U_rep),
        'grad_total': grad_total,
    }


def apf_evasive_gradient_3d(pos, hunter_positions, bound,
                            k_rep_hunter=1.5, k_att_center=0.2, k_wall=0.3,
                            d0_hunter=0.6, d0_wall=0.3):
    """
    Compute 3D APF for the TARGET (evasive) UAV.
    Attraction = away from nearest hunter (escape direction).
    Repulsion = from walls + toward center.

    Returns same dict structure as apf_gradient_3d.
    """
    pos = np.array(pos, dtype=float)
    center = np.array([bound/2., bound/2., bound/2.])

    # "Attraction" for target = escape force (away from nearest hunter)
    f_att = np.zeros(3, dtype=float)
    U_att = 0.0
    for h_p in hunter_positions:
        h_p = np.array(h_p, dtype=float)
        diff_h = pos - h_p
        d_h = np.linalg.norm(diff_h) + 1e-9
        if d_h < d0_hunter:
            inv_d = 1.0 / d_h
            inv_d0 = 1.0 / d0_hunter
            term = inv_d - inv_d0
            U_att += 0.5 * k_rep_hunter * term * term
            dU_dd = k_rep_hunter * term * (-1.0 / (d_h * d_h))
            f_att += -dU_dd * (diff_h / (d_h + 1e-9))

    # "Repulsion" for target = center attraction + wall repulsion
    f_rep = np.zeros(3, dtype=float)
    U_rep = 0.0

    # Weak attractive to center (treated as repulsion from walls)
    diff_c = pos - center
    d_c = np.linalg.norm(diff_c) + 1e-9
    f_rep += -k_att_center * diff_c  # toward center
    U_rep += 0.5 * k_att_center * d_c * d_c

    # Wall repulsion
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
            U_rep += 0.5 * k_wall * term * term
            dU_dd = k_wall * term * (-1.0 / (d_w * d_w))
            f_rep += -dU_dd * normal

    f_att_dir = f_att / (np.linalg.norm(f_att) + 1e-9) if np.linalg.norm(f_att) > 1e-9 else np.zeros(3)
    f_rep_dir = f_rep / (np.linalg.norm(f_rep) + 1e-9) if np.linalg.norm(f_rep) > 1e-9 else np.zeros(3)

    U_total = U_att + U_rep
    grad_total = -f_att - f_rep

    return {
        'f_att_dir': f_att_dir,
        'f_rep_dir': f_rep_dir,
        'U_total': float(U_total),
        'U_att': float(U_att),
        'U_rep': float(U_rep),
        'grad_total': grad_total,
    }


# =============================================================================
# 3D Velocity Obstacle (VO) Cone Functions (方案三)
# =============================================================================

def vo_compute_cone_3d(p_self, p_other, r_self, r_other):
    """
    Compute a single 3D Velocity Obstacle cone for the collision pair
    (self, other) with combined radius R = r_self + r_other.

    VO cone definition (Fiorini & Shiller 1998):
      VO_A,B(tau) = { v | exists t in [0, tau]: p_self + v*t in D(p_other + v_other*t, R) }
    For pure geometry we use truncated cone with apex at p_other, axis = (p_self - p_other).
    For the 3D cone:
      - axis direction  d = normalize(p_self - p_other)
      - half-angle     alpha = arcsin(R / ||p_self-p_other||),  clamped to [eps, pi/2-eps]
      - distance       dist  = ||p_self - p_other||

    Returns: dict{
        'axis': unit vector d pointing from other -> self (cone symmetry axis),
        'cos_half_angle': cos(alpha),   — the cone membership threshold for cosine of angle,
        'sin_half_angle': sin(alpha),
        'half_angle_rad': alpha,
        'distance':       ||p_self - p_other||,
        'R_combined':     r_self + r_other,
        'valid':          True if the cone exists (dist > R_combined);
                          False if already colliding/overlapping — caller should handle separately
    }
    """
    p_self = np.array(p_self, dtype=float)
    p_other = np.array(p_other, dtype=float)
    R = float(r_self) + float(r_other)

    diff = p_self - p_other
    dist = np.linalg.norm(diff) + 1e-9
    axis = diff / dist

    # half-angle alpha = arcsin(R/dist). When dist <= R we are overlapping -> no cone geometry.
    if dist <= R:
        alpha = np.pi / 2.0 - 1e-3  # wide cone as fallback (everything is VO)
        valid = False
    else:
        ratio = R / dist
        ratio = np.clip(ratio, 0.0, 1.0 - 1e-6)
        alpha = np.arcsin(ratio)
        valid = True

    return {
        'axis': axis,
        'cos_half_angle': float(np.cos(alpha)),
        'sin_half_angle': float(np.sin(alpha)),
        'half_angle_rad': float(alpha),
        'distance': float(dist),
        'R_combined': R,
        'valid': valid,
    }


def vo_cone_contains(cone, v_rel):
    """
    Check whether relative velocity v_rel = v_self - v_other is strictly inside
    the 3D VO cone (apex at origin, axis=cone.axis, half-angle = cone.half_angle_rad).

    Membership test:
      let u = normalize(v_rel)
      if dot(u, cone.axis) > cos(alpha)  → v_rel is inside cone (collision velocity)
    Returns: (inside_flag: bool, cos_angle: float)
    """
    v_rel = np.array(v_rel, dtype=float)
    vnorm = np.linalg.norm(v_rel)
    if vnorm < 1e-9:
        # zero relative velocity → static: if overlapping we're colliding, but
        # geometrically a 0 vector points nowhere; treat as inside for safety.
        return (True if not cone['valid'] else False), 1.0
    u = v_rel / vnorm
    cos_angle = float(np.dot(u, cone['axis']))
    return (cos_angle > cone['cos_half_angle']), cos_angle


def vo_project_outside_cone(cone, v_rel):
    """
    Project a 3D relative velocity vector v_rel onto the closest point on the
    VO cone BOUNDARY (outside of cone if it was inside). Returns new v_rel_proj.

    Geometry:
      - v_rel inside cone → rotate it to the cone boundary in the plane
        spanned by (cone.axis, v_rel) such that angle(v_proj, axis) == alpha.
      - v_rel already outside or on boundary → return unchanged.

    Projection method (closed form, preserves ||v_rel|| when possible):
      1. let e1 = cone.axis.
      2. form e2 = normalize(v_rel - dot(v_rel,e1)*e1)  (perpendicular to axis,
         aligned with v_rel component off-axis). If v_rel is parallel to axis,
         pick any perpendicular (e.g. first nonzero of standard basis).
      3. v_proj = |v_rel| * (cos(alpha) * e1 + sin(alpha) * e2)

    The caller then computes v_self_proj = v_other + v_rel_proj.
    """
    v_rel = np.array(v_rel, dtype=float)
    vnorm = np.linalg.norm(v_rel)
    if vnorm < 1e-9:
        # zero vel: rotate to just outside cone (use e2 direction with small step)
        e1 = cone['axis']
        e2 = _any_perp(e1)
        alpha = cone['half_angle_rad']
        # a tiny step along cone boundary:
        return 1e-4 * (np.cos(alpha) * e1 + np.sin(alpha) * e2)

    inside, _ = vo_cone_contains(cone, v_rel)
    if not inside:
        return v_rel.copy()

    e1 = cone['axis']
    e2_raw = v_rel - np.dot(v_rel, e1) * e1
    e2norm = np.linalg.norm(e2_raw)
    if e2norm < 1e-9:
        # v_rel is exactly along cone axis (trivially inside since alpha>0).
        # Use arbitrary perpendicular.
        e2 = _any_perp(e1)
    else:
        e2 = e2_raw / e2norm

    alpha = cone['half_angle_rad']
    # Keep magnitude, angle = alpha from axis (on cone boundary = just outside membership
    # condition if we use >= membership; we project to slightly > cos(alpha) boundary side)
    v_proj = vnorm * (np.cos(alpha) * e1 + np.sin(alpha) * e2)
    # Tiny epsilon safety offset to guarantee "outside" membership:
    cos_angle_new = float(np.dot(v_proj / (np.linalg.norm(v_proj) + 1e-9), e1))
    eps = 1e-6
    if cos_angle_new >= cone['cos_half_angle']:
        # push slightly more off-axis (by increasing sin component)
        beta = alpha + eps
        beta = min(beta, np.pi / 2.0 - 1e-6)
        v_proj = vnorm * (np.cos(beta) * e1 + np.sin(beta) * e2)
    return v_proj


def _any_perp(v):
    """Return a unit vector perpendicular to 3-vector v."""
    v = np.asarray(v, dtype=float)
    if abs(v[0]) < abs(v[1]) and abs(v[0]) < abs(v[2]):
        tmp = np.array([0.0, -v[2], v[1]])
    elif abs(v[1]) < abs(v[2]):
        tmp = np.array([-v[2], 0.0, v[0]])
    else:
        tmp = np.array([-v[1], v[0], 0.0])
    n = np.linalg.norm(tmp)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return tmp / n


def vo_aggregate_obstacles(pos_self, vel_self, obstacles, other_agents,
                           r_self, r_other_dynamic, r_obstacle_static):
    """
    Aggregate all velocity obstacles (dynamic + static) for one agent and
    apply projection to get a safe action velocity.

    For static obstacles (zero velocity) we still use VO with v_other = 0.
    For walls we use per-face plane projection (not cone) as additional safety.

    Args:
      pos_self, vel_self: current 3D position / velocity of self (pre-action input).
      obstacles: list of {'position':[x,y,z], 'radius':r} static spheres.
      other_agents: list of (pos_other, vel_other, r_override_or_None) for dynamic agents.
      r_self: self radius
      r_other_dynamic: default dynamic-agent radius
      r_obstacle_static: default static-obstacle radius override (or use per obstacle dict)

    Returns: (vel_safe_projected, cone_observations_list)
      vel_safe_projected: velocity after one-pass VO+wall projection
      cone_observations_list: list of 6-d observation tuples per top-k cone
                              (axis_x, axis_y, axis_z, cos_half_angle, dist_norm, in_vo_flag)
    """
    pos_self = np.array(pos_self, dtype=float)
    vel_self = np.array(vel_self, dtype=float)
    cones_meta = []  # (cone, v_other, priority_key)
    R_bdy = 1.0  # cube bound side assumed external; will be handled separately

    # (1) dynamic agents: high priority
    for item in other_agents:
        if len(item) == 3:
            po, vo, rr = item
        else:
            po, vo = item
            rr = r_other_dynamic
        cone = vo_compute_cone_3d(pos_self, po, r_self, rr)
        # priority = inverse distance (closer = more important) + dynamic bump
        priority = 1.0 / (cone['distance'] + 1e-6) + 2.0
        cones_meta.append((cone, np.array(vo, dtype=float), priority, 'dynamic'))

    # (2) static obstacles
    for obs in obstacles:
        po = np.array(obs['position'], dtype=float)
        rr = float(obs.get('radius', r_obstacle_static))
        cone = vo_compute_cone_3d(pos_self, po, r_self, rr)
        priority = 1.0 / (cone['distance'] + 1e-6)
        cones_meta.append((cone, np.zeros(3), priority, 'static'))

    # Sort by priority (descending) — apply closest/dynamic first
    cones_meta.sort(key=lambda x: -x[2])

    # (3) Apply projection sequentially (one-pass). Re-check membership after each step
    v_cur = vel_self.copy()
    top_k_for_obs = []  # save first k cones for agent observation
    K_OBS = 3
    for cone, v_other, prio, tag in cones_meta:
        v_rel = v_cur - v_other
        inside, cos_a = vo_cone_contains(cone, v_rel)
        in_flag = 1.0 if inside else 0.0
        if len(top_k_for_obs) < K_OBS:
            dist_norm = float(np.clip(cone['distance'] / 3.0, 0.0, 1.0))  # / L ~= sqrt(3)*1 ~= 3 for norm
            top_k_for_obs.append((cone, dist_norm, in_flag))
        if inside:
            v_rel_proj = vo_project_outside_cone(cone, v_rel)
            v_cur = v_other + v_rel_proj

    # (4) Observations: top-K cones flattened
    obs_vec = []
    for cone, d_norm, in_flag in top_k_for_obs:
        obs_vec.extend([float(cone['axis'][0]), float(cone['axis'][1]), float(cone['axis'][2]),
                        float(cone['cos_half_angle']), d_norm, in_flag])
    # pad if less than K_OBS cones
    while len(obs_vec) < K_OBS * 6:
        obs_vec.extend([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    return v_cur, np.array(obs_vec, dtype=float)
