import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from .defs import System, TrajectoryResult, SystemResult, StatePair, SystemCache
from collections import namedtuple
import logging
import time
from numba import jit
import itertools
import os
import math
import concurrent.futures


Particle = namedtuple("Particle", ["mass", "is_fixed"])
# Particle numbers are zero indexed
Edge = namedtuple("Edge", ["a", "b", "spring_const", "rest_length"])

ParticleTrajectoryResult = namedtuple("ParticleTrajectoryResult",
                                      ["q", "p",
                                       "dq_dt", "dp_dt",
                                       "t_steps",
                                       "p_noiseless", "q_noiseless",
                                       "masses", "edge_indices", "fixed_mask",
                                       "fixed_mask_qp"])


spring_mesh_cache = SystemCache()


@jit(nopython=True)
def batch_outer(v, w):
    res = np.empty((v.shape[0], v.shape[1], w.shape[1]), dtype=v.dtype)
    for i in range(v.shape[0]):
        res[i] = np.outer(v[i], w[i])
    return res


class SpringMeshSystem(System):
    def __init__(self, n_dims, particles, edges, vel_decay, _newton_iter=True):
        super().__init__()
        self.particles = particles
        self.edges = edges
        self.n_dims = n_dims
        assert self.n_dims == 2
        self.n_particles = len(particles)
        self.masses = np.array([p.mass for p in self.particles], dtype=np.float64)
        self.masses.setflags(write=False)
        self.fixed_mask = np.array([p.is_fixed for p in self.particles], dtype=np.bool)
        self.fixed_mask.setflags(write=False)
        self.fixed_idxs = self.fixed_mask.nonzero()[0]
        self.fixed_idxs.setflags(write=False)
        self.viscosity_constant = vel_decay
        # Gather other data
        self.edge_indices = np.array([(e.a, e.b) for e in self.edges] +
                                     [(e.b, e.a) for e in self.edges], dtype=np.int64).T
        self.edge_indices.setflags(write=False)
        self.spring_consts = np.expand_dims(np.array([e.spring_const for e in self.edges] +
                                                     [e.spring_const for e in self.edges], dtype=np.float64),
                                            0)
        self.spring_consts.setflags(write=False)
        self.rest_lengths = np.expand_dims(np.array([e.rest_length for e in self.edges] +
                                                    [e.rest_length for e in self.edges], dtype=np.float64),
                                           0)
        self.rest_lengths.setflags(write=False)
        self.row_coords = np.concatenate([self.edge_indices[0], self.edge_indices[0]])
        self.col_coords = np.concatenate([np.repeat(0, 2 * len(edges)),
                                          np.repeat(1, 2 * len(edges))])
        # Compute the update matrices
        self.mass_mults = np.tile(np.expand_dims(self.masses, 1), (1, self.n_dims)).reshape((-1,))
        self.mass_mults.setflags(write=False)
        self._mass_matrix_sparse = sparse.diags(self.mass_mults)
        assert self._mass_matrix_sparse.shape == (self.n_particles * self.n_dims, self.n_particles * self.n_dims)
        # Build the stiffness matrix
        stiff_mat_parts = np.empty((self.n_particles, self.n_particles), dtype=object)
        for edge in self.edges:
            m_aa = np.diag([edge.spring_const] * self.n_dims)
            m_ab = np.diag([-1 * edge.spring_const] * self.n_dims)
            m_ba = m_ab
            m_bb = m_aa
            for (ai, bi) in itertools.product((edge.a, edge.b), repeat=2):
                if stiff_mat_parts[ai, bi] is None:
                    stiff_mat_parts[ai, bi] = np.zeros((self.n_dims, self.n_dims))
            stiff_mat_parts[edge.a, edge.a] += m_aa
            stiff_mat_parts[edge.a, edge.b] += m_ab
            stiff_mat_parts[edge.b, edge.a] += m_ba
            stiff_mat_parts[edge.b, edge.b] += m_bb
        stiff_mat = sparse.bmat(stiff_mat_parts)
        assert stiff_mat.shape == (self.n_particles * self.n_dims, self.n_particles * self.n_dims)
        # Compute the selection matrix for non-fixed vertices
        n_non_fixed = self.n_particles - np.count_nonzero(self.fixed_mask)
        unfixed_mask_parts = np.empty((n_non_fixed, self.n_particles), dtype=object)
        for i, j in enumerate(np.nonzero(np.logical_not(self.fixed_mask))[0]):
            unfixed_mask_parts[i, j] = np.eye(self.n_dims, dtype=np.bool)
        for i in range(unfixed_mask_parts.shape[0]):
            if unfixed_mask_parts[i, 0] is None:
                unfixed_mask_parts[i, 0] = np.zeros((self.n_dims, self.n_dims), dtype=np.bool)
        for i in range(unfixed_mask_parts.shape[1]):
            if unfixed_mask_parts[0, i] is None:
                unfixed_mask_parts[0, i] = np.zeros((self.n_dims, self.n_dims), dtype=np.bool)
        unfixed_mask_mat = sparse.bmat(unfixed_mask_parts)


        #unfixed_mask_mat.setflags(write=False)
        assert unfixed_mask_mat.shape == (n_non_fixed * self.n_dims, self.n_particles * self.n_dims)
        # Store system matrices
        self._select_matrix = unfixed_mask_mat
        _, self._select_idx_in = self._select_matrix.nonzero()
        self._stiff_mat = stiff_mat
        # Set up support functions for computing derivatives
        edge_indices = self.edge_indices
        n_particles = self.n_particles
        viscosity_constant = self.viscosity_constant
        n_dims = self.n_dims
        spring_consts = self.spring_consts
        rest_lengths = self.rest_lengths
        fixed_mask = self.fixed_mask
        fixed_idxs = self.fixed_mask
        masses = self.masses
        masses_expanded = np.expand_dims(masses, axis=(0, -1))
        masses_repeats = np.tile(np.expand_dims(masses, -1), (1, n_dims)).reshape((-1, ))
        M_inv = np.reciprocal(masses_repeats)
        @jit(nopython=True, fastmath=False)
        def gather_forces(edge_forces, out):
            for i in range(edge_indices.shape[1]):
                a = edge_indices[0, i]
                out[:, a] += edge_forces[:, i]
        @jit(nopython=True, fastmath=False)
        def compute_forces(q, q_dot):
            q = q.reshape((-1, n_particles, n_dims))
            q_dot = q_dot.reshape((-1, n_particles, n_dims))
            # Compute length of each spring and "diff" directions of the forces
            diffs = q[:, edge_indices[0], :] - q[:, edge_indices[1], :]
            lengths = np.sqrt((diffs ** 2).sum(axis=-1))
            # Compute forces
            edge_forces = (np.expand_dims(-1 * spring_consts * (lengths - rest_lengths) / lengths, axis=-1) * diffs)
            # Gather forces for each of their "lead" particles
            forces = np.zeros(shape=(q.shape[0], n_particles, n_dims), dtype=q.dtype)
            gather_forces(edge_forces=edge_forces, out=forces)
            forces -= viscosity_constant * q_dot
            #forces += np.random.randn(*forces.shape) * 0.01
            # Mask forces on fixed particles
            forces[:, fixed_idxs, :] = 0
            return forces
        self.compute_forces = compute_forces
        # Set up free derivative function
        @jit(nopython=True, fastmath=False)
        def derivative(q, p):
            orig_q_shape = q.shape
            orig_p_shape = p.shape
            q_dot = M_inv * p
            q = q.reshape((-1, n_particles, n_dims))
            p = p.reshape((-1, n_particles, n_dims))
            # Compute action of forces on each particle
            forces = compute_forces(q=q, q_dot=q_dot)
            # Update positions
            pos = (1 / masses_expanded) * p
            pos[:, fixed_idxs, :] = 0
            q_out = pos.reshape(orig_q_shape)
            p_out = forces.reshape(orig_p_shape)
            return q_out, p_out
        self.derivative = derivative

        # Configure functions for Newton iteration
        if not _newton_iter:
            return
        jac_idx_arr = np.arange(n_particles * n_dims).reshape((n_particles, n_dims))

        jac_coo_row = []
        jac_coo_col = []
        jac_coo_zero_mask = []
        _fixed_lookup = np.repeat(fixed_mask, n_dims)
        for i in range(edge_indices.shape[-1]):
            a = edge_indices[0, i]
            b = edge_indices[1, i]
            new_rows = [jac_idx_arr[a][0], jac_idx_arr[a][0], jac_idx_arr[a][1], jac_idx_arr[a][1]]
            jac_coo_row.extend(new_rows)
            jac_coo_col.extend([jac_idx_arr[b][0], jac_idx_arr[b][1], jac_idx_arr[b][0], jac_idx_arr[b][1]])
            jac_coo_zero_mask.extend([_fixed_lookup[i] for i in new_rows])
        jac_coo_row = np.array(jac_coo_row, dtype=np.uint64)
        jac_coo_col = np.array(jac_coo_col, dtype=np.uint64)
        jac_coo_zero_mask = np.array(jac_coo_zero_mask, dtype=np.bool)

        arr_size = n_particles * n_dims
        eye_n_dims = np.tile(np.expand_dims(np.eye(n_dims), axis=0), (2*len(self.edges), 1, 1))
        _jac_b_diag = np.ones(arr_size)
        _jac_b_diag[_fixed_lookup] = 0
        jac_b = -viscosity_constant * sparse.diags(_jac_b_diag, shape=(arr_size, arr_size))

        I_zero = sparse.hstack([sparse.eye(arr_size), sparse.coo_matrix((arr_size, arr_size))])
        zero_I = sparse.hstack([sparse.coo_matrix((arr_size, arr_size)), sparse.eye(arr_size)])
        M_inv_spdiag = sparse.diags(M_inv, shape=(M_inv.shape[0], M_inv.shape[0]))

        eye_n_dims.setflags(write=False)

        def _force_grad(q, p):
            q = q.reshape((n_particles, n_dims))
            p = p.reshape((n_particles, n_dims))

            diff = q[edge_indices[0], :] - q[edge_indices[1], :]
            norm2 = np.sqrt((diff ** 2).sum(axis=-1))
            lead_coeffs = np.expand_dims(np.expand_dims(spring_consts[0], axis=-1), axis=-1)

            term1_scalar = np.expand_dims(np.expand_dims(norm2 - rest_lengths[0], axis=-1), axis=-1)
            term1_deriv_vec = diff / np.expand_dims(norm2, axis=-1)
            term2_vec = diff / np.expand_dims(norm2, axis=-1)
            outer_prod = batch_outer(diff, diff)

            term2_a = eye_n_dims / np.expand_dims(np.expand_dims(norm2, axis=-1), axis=-1)
            term2_b = outer_prod/np.expand_dims(np.expand_dims(norm2**3, axis=-1), axis=-1)
            term2_deriv_mat = term2_a - term2_b
            term_ab = (lead_coeffs * (batch_outer(term1_deriv_vec, term2_vec)) +
                       lead_coeffs * (term1_scalar * term2_deriv_mat))
            # Store results from term_ab
            term_ab_sparse = term_ab.reshape((-1, ))
            term_ab_sparse[jac_coo_zero_mask] = 0
            jac_sparse = sparse.coo_matrix((term_ab_sparse, (jac_coo_row, jac_coo_col)), shape=(arr_size, arr_size))
            res_sparse = sparse.hstack([jac_sparse, jac_b])
            return res_sparse

        @jit(nopython=True)
        def _newton_func_val(q_prev, q_dot_prev, q_next, q_dot_next, dt):
            forces = compute_forces(q_next, q_dot_next).reshape((-1, ))
            q_val = q_next - q_prev - dt * q_dot_prev - dt**2 * M_inv * forces
            q_dot_val = q_dot_next - q_dot_prev - dt * M_inv * forces
            return np.concatenate((q_val, q_dot_val), axis=-1)

        def _newton_func_jac(q_prev, q_dot_prev, q_next, q_dot_next, dt):
            minv_jf = M_inv_spdiag @ _force_grad(q_next, q_dot_next)
            q_rows = I_zero - dt**2 * minv_jf
            q_dot_rows = zero_I - dt * minv_jf
            return sparse.vstack([q_rows, q_dot_rows])

        @jit(nopython=True)
        def _bdf_2_func(q_prev, q_prev_prev, q_dot_prev, q_dot_prev_prev, q_next, q_dot_next, dt):
            forces = compute_forces(q_next, q_dot_next).reshape((-1, ))
            q_val = q_next - (4 / 3) * q_prev + (1 / 3) * q_prev_prev - (2 / 3) * dt * q_dot_next
            q_dot_val = q_dot_next - (4 / 3) * q_dot_prev + (1 / 3) * q_dot_prev_prev - (2 / 3) * dt * M_inv * forces
            return np.concatenate((q_val, q_dot_val), axis=-1)

        def _bdf_2_jac(q_prev, q_dot_prev, q_next, q_dot_next, dt):
            minv_jf = M_inv_spdiag @ _force_grad(q_next, q_dot_next)
            q_rows = I_zero - ((2 / 3) * dt)**2 * minv_jf
            q_dot_rows = zero_I - (2 / 3) * dt * minv_jf
            return sparse.vstack([q_rows, q_dot_rows])

        # Do the newton iterations
        def compute_next_step(q_prev, q_dot_prev, dt):
            split_idx = n_dims * n_particles
            q_prev = q_prev.reshape((-1, ))
            q_dot_prev = q_dot_prev.reshape((-1, ))
            q_next = q_prev.copy()
            q_dot_next = q_dot_prev.copy()
            val = _newton_func_val(q_prev, q_dot_prev, q_next, q_dot_next, dt)
            num_iter = 0
            while np.linalg.norm(val) > 1e-12:
                val = _newton_func_val(q_prev, q_dot_prev, q_next, q_dot_next, dt)
                jac = _newton_func_jac(q_prev, q_dot_prev, q_next, q_dot_next, dt)
                jac_inv_prod = sp_linalg.spsolve(jac, val)
                q_incr = jac_inv_prod[:split_idx]
                q_dot_incr = jac_inv_prod[split_idx:]
                q_next = q_next - q_incr
                q_dot_next = q_dot_next - q_dot_incr
                num_iter += 1
                if num_iter > 50:
                    break
            return q_next, q_dot_next

        # Do the newton iterations
        def compute_next_step_bdf_2(q_prev, q_prev_prev, q_dot_prev, q_dot_prev_prev, dt):
            split_idx = n_dims * n_particles
            q_prev = q_prev.reshape((-1, ))
            q_prev_prev = q_prev_prev.reshape((-1, ))
            q_dot_prev = q_dot_prev.reshape((-1, ))
            q_dot_prev_prev = q_dot_prev_prev.reshape((-1, ))
            q_next = q_prev.copy()
            q_dot_next = q_dot_prev.copy()
            val = _bdf_2_func(q_prev, q_prev_prev, q_dot_prev, q_dot_prev_prev, q_next, q_dot_next, dt)
            num_iter = 0
            while np.linalg.norm(val) > 1e-12:
                val = _bdf_2_func(q_prev, q_prev_prev, q_dot_prev, q_dot_prev_prev, q_next, q_dot_next, dt)
                jac = _bdf_2_jac(q_prev, q_dot_prev, q_next, q_dot_next, dt)
                jac_inv_prod = sp_linalg.spsolve(jac, val)
                q_incr = jac_inv_prod[:split_idx]
                q_dot_incr = jac_inv_prod[split_idx:]
                q_next = q_next - q_incr
                q_dot_next = q_dot_next - q_dot_incr
                num_iter += 1
                if num_iter > 50:
                    break
            return q_next, q_dot_next

        def back_euler(q0, p0, dt, out_q, out_p):
            q = q0.reshape((-1, ))
            q_dot = M_inv * p0.reshape((-1, ))
            for i in range(out_q.shape[0]):
                out_q[i] = q
                out_p[i] = masses_repeats * q_dot
                q, q_dot = compute_next_step(q, q_dot, dt)
        self.back_euler = back_euler

        def bdf_2(q0, p0, dt, out_q, out_p):
            q_prev = q0.reshape((-1, ))
            q_dot_prev = M_inv * p0.reshape((-1, ))
            out_q[0] = q_prev
            out_p[0] = masses_repeats * q_dot_prev
            q, q_dot = compute_next_step(q_prev, q_dot_prev, dt)
            for i in range(1, out_q.shape[0]):
                out_q[i] = q
                out_p[i] = masses_repeats * q_dot
                tmp = (q, q_dot)
                q, q_dot = compute_next_step_bdf_2(q, q_prev, q_dot, q_dot_prev, dt)
                q_prev, q_dot_prev = tmp
        self.bdf_2 = bdf_2

    def _args_compatible(self, n_dims, particles, edges, vel_decay):
        return (self.n_dims == n_dims and
                self.particles == particles and
                set(self.edges) == set(edges) and
                self.viscosity_constant == vel_decay)

    def hamiltonian(self, q, p):
        return np.zeros([q.shape[0], q.shape[1]])

    def _compute_next_step(self, q, q_dot, time_step_size, mat_solver):
        # Input states are (n_particle, n_dim)
        forces_orig = self.compute_forces(q=q, q_dot=q_dot)[0]
        forces = forces_orig.reshape((-1,))
        q = q.reshape((-1, ))
        q_dot = q_dot.reshape((-1, ))
        known = ((self.mass_mults * q_dot)[self._select_idx_in] + (time_step_size * (forces[self._select_idx_in])))
        # Two of the values to return
        q_dot_hat_next = mat_solver(known)
        q_next = q.copy()
        q_next[self._select_idx_in] += time_step_size * q_dot_hat_next
        # Reshape
        q_dot_next = np.zeros_like(q_dot)
        q_dot_next[self._select_idx_in] = q_dot_hat_next
        q_next = q_next.reshape((self.n_particles, self.n_dims))
        # Get the p values to return
        p = (self.mass_mults * q_dot_next).reshape((self.n_particles, self.n_dims))
        p[self.fixed_idxs] = 0
        q_dot_next = q_dot_next.reshape((self.n_particles, self.n_dims))
        return q_next, q_dot_next, p, forces_orig

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size,
                            subsample=1, noise_sigma=0.0):
        # Check shapes of inputs
        if (q0.shape != (self.n_particles, self.n_dims)) or (p0.shape != (self.n_particles, self.n_dims)):
            raise ValueError("Invalid input shape for particle system")

        t_eval = np.arange(num_time_steps) * time_step_size

        # Process arguments for subsampling
        num_steps = num_time_steps * subsample
        orig_time_step_size = time_step_size
        time_step_size = time_step_size / subsample

        # Compute updates using explicit Euler
        # compute update matrices
        mat_unknown = self._select_matrix @ (self._mass_matrix_sparse - (time_step_size ** 2) * self._stiff_mat) @ self._select_matrix.T
        mat_solver = sp_linalg.factorized(mat_unknown.tocsc())

        init_vel = np.zeros_like(q0)
        for i, part in enumerate(self.particles):
            init_vel[i] = (1/part.mass) * p0[i]

        qs = [q0]
        q_dots = [init_vel]
        ps = [p0]
        p_dots = [self.compute_forces(q=q0, q_dot=p0)[0]]
        q = q0.copy()
        q_dot = p0.copy()

        for i, part in enumerate(self.particles):
            q_dot[i] /= part.mass
        for step_idx in range(1, num_steps):
            q, q_dot, p, _p_dot_next = self._compute_next_step(q=q, q_dot=q_dot, time_step_size=time_step_size,
                                                               mat_solver=mat_solver)
            if step_idx % subsample == 0:
                p_dot = self.compute_forces(q=q, q_dot=p)[0]
                qs.append(q)
                q_dots.append(q_dot)
                ps.append(p)
                p_dots.append(p_dot)

        qs = np.stack(qs).reshape(num_time_steps, self.n_particles, self.n_dims)
        ps = np.stack(ps).reshape(num_time_steps, self.n_particles, self.n_dims)
        dq_dt = np.stack(q_dots).reshape(num_time_steps, self.n_particles, self.n_dims)
        dp_dt = np.stack(p_dots).reshape(num_time_steps, self.n_particles, self.n_dims)

        # Add configured noise
        noise_ps = noise_sigma * np.random.randn(*ps.shape)
        noise_qs = noise_sigma * np.random.randn(*qs.shape)

        qs_noisy = qs + noise_qs
        ps_noisy = ps + noise_ps

        # Gather other data
        edge_indices = np.array([(e.a, e.b) for e in self.edges] +
                                [(e.b, e.a) for e in self.edges], dtype=np.int64).T

        fixed_mask_qp = np.stack([self.fixed_mask, self.fixed_mask], axis=-1)

        return ParticleTrajectoryResult(
            q=qs_noisy,
            p=ps_noisy,
            dq_dt=dq_dt,
            dp_dt=dp_dt,
            t_steps=t_eval,
            q_noiseless=qs,
            p_noiseless=ps,
            masses=self.masses,
            edge_indices=edge_indices,
            fixed_mask=self.fixed_mask,
            fixed_mask_qp=fixed_mask_qp,
        )


def system_from_records(n_dims, particles, edges, vel_decay):
    parts = []
    edgs = []
    for pdef in particles:
        parts.append(
            Particle(mass=pdef["mass"],
                     is_fixed=pdef["is_fixed"]))
    for edef in edges:
        edgs.append(
            Edge(a=edef["a"],
                 b=edef["b"],
                 spring_const=edef["spring_const"],
                 rest_length=edef["rest_length"]))
    cached_sys = spring_mesh_cache.find(n_dims=n_dims,
                                        particles=parts,
                                        edges=edgs,
                                        vel_decay=vel_decay)
    if cached_sys is not None:
        return cached_sys
    else:
        new_sys = SpringMeshSystem(n_dims=n_dims,
                                   particles=parts,
                                   edges=edgs,
                                   vel_decay=vel_decay)
        spring_mesh_cache.insert(new_sys)
        return new_sys


def make_enforce_boundary_function(trajectory):
    # Prep fixed mask for enforcement
    fm_q = trajectory.fixed_mask_q[0].cpu().numpy().reshape((-1, ))
    fm_q.setflags(write=False)
    base_q = trajectory.q[0, 0].cpu().numpy().reshape((-1, ))[fm_q]
    base_q.setflags(write=False)
    fm_p = trajectory.fixed_mask_p[0].cpu().numpy().reshape((-1, ))
    fm_p.setflags(write=False)

    @jit(nopython=True)
    def spring_mesh_boundary_condition(q, p, t):
        q = q.copy()
        p = p.copy()
        q[:, fm_q] = base_q
        p[:, fm_p] = 0
        return q, p

    return spring_mesh_boundary_condition


def _generate_data_worker(i, traj_def, vel_decay):
    traj_name = f"traj_{i:05}"
    base_logger = logging.getLogger("spring-mesh")
    traj_logger = base_logger.getChild(traj_name)
    traj_logger.info(f"Generating trajectory {traj_name}")

    # Create the trajectory
    particle_defs = traj_def["particles"]
    spring_defs = traj_def["springs"]
    num_time_steps = traj_def["num_time_steps"]
    time_step_size = traj_def["time_step_size"]
    noise_sigma = traj_def.get("noise_sigma", 0)
    subsample = int(traj_def.get("subsample", 1))

    # Split particles and springs into components
    q0 = []
    particles = []
    edges = []
    for pdef in particle_defs:
        particles.append(
            Particle(mass=pdef["mass"],
                     is_fixed=pdef["is_fixed"]))
        q0.append(np.array(pdef["position"]))
    for edef in spring_defs:
        edges.append(
            Edge(a=edef["a"],
                 b=edef["b"],
                 spring_const=edef["spring_const"],
                 rest_length=edef["rest_length"]))
    q0 = np.stack(q0).astype(np.float64)
    p0 = np.zeros_like(q0)

    n_dims = q0.shape[-1]
    n_particles = len(particle_defs)

    system = spring_mesh_cache.find(n_dims=n_dims, particles=particles,
                                    edges=edges, vel_decay=vel_decay)
    if system is None:
        system = SpringMeshSystem(n_dims=n_dims, particles=particles,
                                  edges=edges, vel_decay=vel_decay, _newton_iter=False)
        spring_mesh_cache.insert(system)

    traj_gen_start = time.perf_counter()
    traj_result = system.generate_trajectory(q0=q0,
                                             p0=p0,
                                             num_time_steps=num_time_steps,
                                             time_step_size=time_step_size,
                                             subsample=subsample,
                                             noise_sigma=noise_sigma)
    traj_gen_elapsed = time.perf_counter() - traj_gen_start
    traj_logger.info(f"Generated {traj_name} in {traj_gen_elapsed} sec")

    trajectories_update = {
        f"{traj_name}_p": traj_result.p,
        f"{traj_name}_q": traj_result.q,
        f"{traj_name}_dqdt": traj_result.dq_dt,
        f"{traj_name}_dpdt": traj_result.dp_dt,
        f"{traj_name}_t": traj_result.t_steps,
        f"{traj_name}_p_noiseless": traj_result.p_noiseless,
        f"{traj_name}_q_noiseless": traj_result.q_noiseless,
        f"{traj_name}_masses": traj_result.masses,
        f"{traj_name}_edge_indices": traj_result.edge_indices,
        f"{traj_name}_fixed_mask": traj_result.fixed_mask,
        f"{traj_name}_fixed_mask_qp": traj_result.fixed_mask_qp,
    }

    trajectory_metadata = {
        "name": traj_name,
        "num_time_steps": num_time_steps,
        "time_step_size": time_step_size,
        "noise_sigma": noise_sigma,
        "field_keys": {
            "p": f"{traj_name}_p",
            "q": f"{traj_name}_q",
            "dpdt": f"{traj_name}_dpdt",
            "dqdt": f"{traj_name}_dqdt",
            "t": f"{traj_name}_t",
            "p_noiseless": f"{traj_name}_p_noiseless",
            "q_noiseless": f"{traj_name}_q_noiseless",
            "masses": f"{traj_name}_masses",
            "edge_indices": f"{traj_name}_edge_indices",
            # Fixed masks
            "fixed_mask": f"{traj_name}_fixed_mask",
            "fixed_mask_p": f"{traj_name}_fixed_mask_qp",
            "fixed_mask_q": f"{traj_name}_fixed_mask_qp",
            "extra_fixed_mask": f"{traj_name}_fixed_mask",
        },
        "timing": {
            "traj_gen_time": traj_gen_elapsed
        }
    }

    if noise_sigma == 0:
        # Deduplicate noiseless trajectory
        del trajectories_update[f"{traj_name}_p_noiseless"]
        del trajectories_update[f"{traj_name}_q_noiseless"]
        traj_keys = trajectory_metadata["field_keys"]
        traj_keys["q_noiseless"] = traj_keys["q"]
        traj_keys["p_noiseless"] = traj_keys["p"]

    return trajectories_update, (i, trajectory_metadata)


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring-mesh")
    else:
        logger = logging.getLogger("spring-mesh")

    trajectory_metadata = []
    trajectories = {}
    vel_decay = system_args.get("vel_decay", 0.0)
    trajectory_defs = system_args["trajectory_defs"]

    # Determine number of cores accessible from this job
    num_cores = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE",
                                   len(os.sched_getaffinity(0))))
    # Limit workers to at most the number of trajectories
    num_tasks = min(num_cores, len(trajectory_defs))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_tasks) as executor:
        futures = []
        for i, traj_def in enumerate(trajectory_defs):
            futures.append(executor.submit(_generate_data_worker, i=i, traj_def=traj_def, vel_decay=vel_decay))

        for future in concurrent.futures.as_completed(futures):
            trajectories_update, traj_meta_update = future.result()
            trajectories.update(trajectories_update)
            trajectory_metadata.append(traj_meta_update)

    # Perform final processing of output
    logger.info("Done generating trajectories")
    trajectory_metadata.sort()
    trajectory_metadata = [d for _i, d in trajectory_metadata]

    particle_records = []
    edge_records = []
    for part in trajectory_defs[0]["particles"]:
        particle_records.append({
            "mass": part["mass"],
            "is_fixed": part["is_fixed"],
        })
    for edge in trajectory_defs[0]["springs"]:
        edge_records.append({
            "a": edge["a"],
            "b": edge["b"],
            "spring_const": edge["spring_const"],
            "rest_length": edge["rest_length"],
        })

    n_particles = len(particle_records)
    n_dims = 2

    return SystemResult(trajectories=trajectories,
                        metadata={
                            "n_grid": n_dims,
                            "n_dim": n_dims,
                            "n_particles": n_particles,
                            "system_type": "spring-mesh",
                            "particles": particle_records,
                            "edges": edge_records,
                            "vel_decay": vel_decay,
                        },
                        trajectory_metadata=trajectory_metadata)
