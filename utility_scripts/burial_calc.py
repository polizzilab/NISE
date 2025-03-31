import torch
import numpy as np
import prody as pr
import scipy.spatial as sp
from scipy.spatial.transform import Rotation as R
from typing import Optional

def plot_alpha_shape(points, triangles, additional_coords=None, no_plot=False):
    """
    Helper function for debugging and visualization of the alphahull shape
    """
    from plotly import graph_objects as go # type: ignore

    # Extract the coordinates of the vertices
    x, y, z = points.T

    # Create a Mesh3D plot using the vertices and the triangles
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        opacity=0.4,
        color='pink'
    )

    # Create a figure and add the mesh
    fig = go.Figure(data=[mesh])

    # Optionally, add the original points as a scatter3d plot for reference
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color='black',
            opacity=0.5
        )
    )
    fig.add_trace(scatter)

    if additional_coords is not None:
        scatter2 = go.Scatter3d(
            x=additional_coords[:, 0],
            y=additional_coords[:, 1],
            z=additional_coords[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                symbol='diamond',
                color='blue',
                opacity=0.5
            )
        )
        fig.add_trace(scatter2)

    # Set the layout of the figure
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis'),
        ),
        title='Alpha Shape (Concave Hull) Visualization'
    )

    # Show the figure
    if not no_plot:
        fig.show()

    return fig


def output_protein_with_hull_betas(bb_frames, sequence_indices, betas):
    out = pr.AtomGroup()
    out.setCoords(bb_frames.reshape(-1, 3))
    out.setNames(frame_order*bb_frames.shape[0]) # type: ignore
    out.setOccupancies([1] * bb_frames.shape[0] * len(frame_order)) # type: ignore
    out.setResnames([aa_idx_to_long[x] for x in sequence_indices for _ in frame_order]) # type: ignore
    out.setElements([x[0] for x in frame_order]*bb_frames.shape[0]) # type: ignore
    out.setResnums([x for x in range(1, bb_frames.shape[0]+1) for _ in frame_order]) # type: ignore
    out.setBetas(betas[:, None].repeat(len(frame_order), axis=1).flatten()) # type: ignore
    return out


def compute_alpha_hull(pos, alpha):
    """
    Grabbed from here:
    https://stackoverflow.com/questions/26303878/alpha-shapes-in-3d

    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    assert pos.shape[0] > 3, "Need at least four points to form a tetrahedron"
    tetra = sp.Delaunay(pos)

    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.simplices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(np.maximum(0, Dx**2 + Dy**2 + Dz**2 - 4*a*c))/( (2 * np.abs(a)) + 1e-6 )

    # Find tetrahedrals
    tetras = tetra.simplices[r<alpha,:]

    # triangles
    tricomb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    triangles = tetras[:,tricomb].reshape(-1,3)
    triangles = np.sort(triangles,axis=1)

    # remove triangles that occur twice, because they are within shapes
    triangles, counts = np.unique(triangles, axis=0, return_counts=True)
    triangles = triangles[counts == 1]

    return triangles


def generate_raycast_seed(points):
    """
    Generate a raycast seed point for a point cloud.

    Parameters:
    - points: numpy array of shape (N, 3)
        The input point cloud.

    Returns:
    - rotated_point: numpy array of shape (3,)
        The rotated point outside the bounding box of the point cloud.
    """
    # Find a point outside the bounding box of the point cloud
    max_abs_coords = np.max(np.abs(points), axis=0)
    outside_point = max_abs_coords + np.random.uniform(25, 100, size=3)  # Adjust the range as needed

    # Apply a random rotation to the point
    rotation = R.random().as_matrix()  # Generate a random 3D rotation matrix
    rotated_point = rotation @ outside_point  # Apply the rotation

    return rotated_point


def moller_trumbore_pytorch(ray_origins, ray_directions, triangles):
    """
    Computes the intersections between rays and triangles using the Moller-Trumbore algorithm.

    Parameters:
    - ray_origins (Tensor): A tensor of shape (N, 3) containing the origins of N rays.
    - ray_directions (Tensor): A tensor of shape (N, 3) containing the normalized direction vectors of N rays.
    - triangles (Tensor): A tensor of shape (M, 3, 3) containing M triangles defined by their vertices.

    Returns:
    - valid_intersections (Tensor): A boolean tensor of shape (N, M) indicating if a ray intersects a triangle.
    - t (Tensor): A tensor of shape (N, M) containing the distance from the ray origin to the intersection point.

    The algorithm returns a boolean tensor indicating whether each ray intersects with each triangle and a tensor
    with the corresponding intersection distances. Intersections are only considered valid if they occur in the
    direction of the ray and within the bounds of the triangle.

    Function generated by GPT-4 Turbo.
    """
    EPSILON = 1e-6  # A small constant to avoid division by zero and floating point errors

    # Calculate edges of the triangle
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]

    # Compute the determinant
    h = torch.cross(ray_directions.unsqueeze(1), edge2.unsqueeze(0), dim=2)
    a = (edge1.unsqueeze(0) * h).sum(-1)
    mask = torch.abs(a) > EPSILON

    # Calculate the inverse determinant
    f = torch.where(mask, 1.0 / a, torch.zeros_like(a))

    # Calculate the distance from the first vertex to the ray origin
    s = ray_origins.unsqueeze(1) - triangles[:, 0].unsqueeze(0)

    # Compute the barycentric coordinate u
    u = f * (s * h).sum(-1)

    # Compute the cross product for the second barycentric coordinate v
    q = torch.cross(s, edge1.unsqueeze(0), dim=2)

    # Compute the barycentric coordinate v and the distance t along the ray to the intersection point
    v = f * (ray_directions.unsqueeze(1) * q).sum(-1)
    t = f * (edge2.unsqueeze(0) * q).sum(-1)

    # Determine if the intersection is valid based on u, v, and t
    valid_intersections = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > EPSILON)

    return valid_intersections, t


def point_inside_mesh_raycast_pytorch(test_points: np.ndarray, pos: np.ndarray, triangles: np.ndarray):
    """
    Determines whether each point in a batch of test points is inside a 3D triangle mesh.

    Parameters:
    - test_points (array): An array of points to test, shape (N, 3).
    - pos (array): An array of vertex positions of the mesh, shape (V, 3).
    - triangles (array): An array of indices that constitute the mesh triangles, shape (T, 3).

    Returns:
    - Tensor: A boolean tensor of shape (N,) indicating True if the point is inside the mesh and False otherwise.

    This function uses the ray casting method to determine if a point is inside a 3D mesh. For each test point, a ray
    is cast in a random direction, and the number of intersections with the mesh is counted. If the count is odd, the
    point is inside; if even, the point is outside.

    Function generated by GPT-4 Turbo.
    """
    # Convert numpy arrays to PyTorch tensors
    pos_tensor = torch.tensor(pos, dtype=torch.float32)
    triangles_tensor = torch.tensor(triangles, dtype=torch.long)
    test_points_tensor = torch.tensor(test_points, dtype=torch.float32)

    # Generate a random point outside the mesh to serve as one end of the ray
    random_outside_point = generate_raycast_seed(pos_tensor.numpy())
    random_outside_point_tensor = torch.tensor(random_outside_point, dtype=torch.float32)

    # Compute the ray directions for all test points
    ray_directions = test_points_tensor - random_outside_point_tensor
    ray_lengths = torch.norm(ray_directions, dim=1)
    ray_directions = ray_directions / ray_lengths[:, None]  # Normalize the ray directions

    # Prepare the array of triangles for PyTorch computation
    triangle_vertices = pos_tensor[triangles_tensor]

    # Compute intersections for all rays and triangles at once using the Moller-Trumbore algorithm
    valid_intersections, t_values = moller_trumbore_pytorch(test_points_tensor, ray_directions, triangle_vertices)

    # Filter the t_values to only include intersections that are within the ray lengths
    valid_t_values = t_values.where(valid_intersections, torch.tensor(float('inf')).to(t_values.device))
    intersection_counts = torch.sum(valid_t_values <= ray_lengths[:, None], dim=1)

    # A point is inside the mesh if the number of triangle intersections with the ray is odd
    return intersection_counts % 2 == 1


def compute_fast_ligand_burial_mask(ca_coords_: np.ndarray, lig_coords_: np.ndarray, alpha: float = 9.0, num_rays: int = 1):

    # Copy the coords to translate non-destructively
    lig_coords = lig_coords_.copy()
    ca_coords = ca_coords_.copy()

    # Shift the coordinates to the origin about CA atom.
    lig_coords -= ca_coords.mean(axis=0)
    ca_coords -= ca_coords.mean(axis=0)

    # Compute the convex hull of the CA atoms
    triangles = compute_alpha_hull(ca_coords, alpha)

    lig_mask = point_inside_mesh_raycast_pytorch(lig_coords, ca_coords, triangles)
    for _ in range(0, num_rays - 1):
        lig_mask = lig_mask & point_inside_mesh_raycast_pytorch(lig_coords, ca_coords, triangles)

    return lig_mask


def compute_fast_alphahull_burial_mask(bb_coords: np.ndarray, lig_coords: Optional[np.ndarray] = None, alpha: float = 9.0, num_rays: int = 1):
    """
    Returns a boolean mask indicating whether each CB atom is buried within the protein structure.
    """
    assert num_rays > 0, "The number of rays must be a positive integer"
    assert alpha > 0, "The alpha value must be a positive number"

    # Copy the backbone coordinates
    bb_frames = bb_coords.copy()

    if lig_coords is not None:
        lig_coords = lig_coords.copy()
        lig_coords -= bb_frames[:, 1].mean(axis=0)

    # Shift the coordinates to the origin about CA atom.
    bb_frames -= bb_frames[:, 1].mean(axis=0)

    # Compute the convex hull of the CA atoms
    points = bb_frames[:, 1]
    cb_points = bb_frames[:, 2]

    # Handle the case where there are not enough points to compute the convex hull with sp.Delaunay
    if points.shape[0] <= 5:
        if lig_coords is None:
            return torch.zeros(cb_points.shape[0]), torch.empty(0, dtype=torch.bool)
        else:
            return torch.zeros(cb_points.shape[0]), torch.zeros(lig_coords.shape[0])

    triangles = compute_alpha_hull(points, alpha)

    # Compute the burial mask
    cb_mask = point_inside_mesh_raycast_pytorch(cb_points, points, triangles)
    for _ in range(0, num_rays - 1):
        cb_mask = cb_mask & point_inside_mesh_raycast_pytorch(cb_points, points, triangles)

    if lig_coords is None:
        return cb_mask, torch.empty(0, dtype=torch.bool)

    lig_mask = point_inside_mesh_raycast_pytorch(lig_coords, points, triangles)
    for _ in range(0, num_rays - 1):
        lig_mask = lig_mask & point_inside_mesh_raycast_pytorch(lig_coords, points, triangles)

    return cb_mask, lig_mask