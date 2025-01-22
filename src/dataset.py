import json
import pathlib
from collections import namedtuple
import numpy as np
import torch
from torch.utils import data
import math

Trajectory = namedtuple("Trajectory", ["name", "q", "p", "q_next", "p_next", "dq_dt", "dp_dt", "u",
                                       "t", "dt", "trajectory_meta",
                                       "q_noiseless", "p_noiseless",
                                       "masses", "edge_index", "vertices",
                                       "fixed_mask_q", "fixed_mask_p",
                                       "extra_fixed_mask", "static_nodes", "sys_id"
                                       ])

class TrajectoryDataset(Dataset):
    """Returns batches of full trajectories.
    dataset[idx] -> a set of snapshots for a full trajectory"""

    def __init__(self, data_dir, seq_len: int = 1, stride: int = 1):
        super().__init__()
        data_dir = pathlib.Path(data_dir)

        self.seq_len = seq_len
        self.stride = stride

        with open(data_dir / "system_meta.json", "r", encoding="utf8") as meta_file:
            metadata = json.load(meta_file)
        self.system = metadata["system"]
        self.system_metadata = metadata["metadata"]
        self._trajectory_meta = metadata["trajectories"]
        self._npz_file = np.load(data_dir / "trajectories.npz")
        
        self._linearize = True

    def linearize(self, arr):
        if not isinstance(arr, torch.Tensor) or not self._linearize:
            return arr
        num_steps = arr.shape[0]
        if arr.dim() >= 3:
            # Flatten dimensions 1 and 2
            return arr.flatten(start_dim=2).transpose(1, 2).reshape(num_steps, -1)
        elif arr.dim() == 2:
            # Transpose and flatten for 2D tensors
            return arr.T.reshape(num_steps, -1)

        return arr  # Return unchanged if none of the conditions apply

    def _extract_seq_data(self, array):
        """Extracts sequence array from the dataset."""
        if array.dim() == 1:
            array = array.unsqueeze(1)
            
        array = self.linearize(array)

        timesteps = array.shape[0]
        seq_list = []
        for i in range(0, timesteps - self.seq_len + 1, self.stride):
            seq_list.append(array[i:i+self.seq_len,:])

        # (Seq., Timesteps, States, Systems)
        return torch.stack(seq_list, dim=1)

    def __getitem__(self, idx):
        meta = self._trajectory_meta[idx]
        name = meta["name"]
        dt = torch.tensor(meta["time_step_size"], dtype=torch.float32)
        sys_id = torch.tensor(int(name.split('_')[1]), dtype=torch.int32)
        # Load arrays
        p = torch.from_numpy(self._npz_file[meta["field_keys"]["p"]])[:-1,:,:]
        q = torch.from_numpy(self._npz_file[meta["field_keys"]["q"]])[:-1,:,:]
        p_next = torch.from_numpy(self._npz_file[meta["field_keys"]["p"]])[1:,:,:]
        q_next = torch.from_numpy(self._npz_file[meta["field_keys"]["q"]])[1:,:,:]
                                                                           
        dp_dt = torch.from_numpy(self._npz_file[meta["field_keys"]["dpdt"]])[:-1,:,:]
        dq_dt = torch.from_numpy(self._npz_file[meta["field_keys"]["dqdt"]])[:-1,:,:]
        t = torch.from_numpy(self._npz_file[meta["field_keys"]["t"]])[:-1]
        u = torch.zeros_like(t)
        # Handle (possibly missing) noiseless data
        if "p_noiseless" in meta["field_keys"] and "q_noiseless" in meta["field_keys"]:
            # We have explicit noiseless data
            p_noiseless = torch.from_numpy(self._npz_file[meta["field_keys"]["p_noiseless"]])[:-1,:,:]
            q_noiseless = torch.from_numpy(self._npz_file[meta["field_keys"]["q_noiseless"]])[:-1,:,:]
        else:
            # Data must already be noiseless
            p_noiseless = torch.from_numpy(self._npz_file[meta["field_keys"]["p"]])[:-1,:,:]
            q_noiseless = torch.from_numpy(self._npz_file[meta["field_keys"]["q"]])[:-1,:,:]
        # Handle (possibly missing) masses
        if "masses" in meta["field_keys"]:
            masses = torch.from_numpy(self._npz_file[meta["field_keys"]["masses"]])
        else:
            num_particles = p.shape[1]
            masses = torch.ones(num_particles)
        if "edge_indices" in meta["field_keys"]:
            edge_index = torch.from_numpy(self._npz_file[meta["field_keys"]["edge_indices"]])
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T
        else:
            edge_index = torch.tensor([])
        if "vertices" in meta["field_keys"]:
            vertices = torch.from_numpy(self._npz_file[meta["field_keys"]["vertices"]])
        else:
            vertices = torch.tensor([])

        # Handle per-trajectory boundary masks
        if "fixed_mask_p" in meta["field_keys"]:
            fixed_mask_p = torch.from_numpy(np.expand_dims(self._npz_file[meta["field_keys"]["fixed_mask_p"]], 0))
        else:
            fixed_mask_p = torch.tensor([[]])
        if "fixed_mask_q" in meta["field_keys"]:
            fixed_mask_q = torch.from_numpy(np.expand_dims(self._npz_file[meta["field_keys"]["fixed_mask_q"]], 0))
        else:
            fixed_mask_q = torch.tensor([[]])
        if "extra_fixed_mask" in meta["field_keys"]:
            extra_fixed_mask = torch.from_numpy(np.expand_dims(self._npz_file[meta["field_keys"]["extra_fixed_mask"]], 0))
        else:
            extra_fixed_mask = torch.tensor([[]])
        if "enumerated_fixed_mask" in meta["field_keys"]:
            static_nodes = torch.from_numpy(np.expand_dims(self._npz_file[meta["field_keys"]["enumerated_fixed_mask"]], 0))
        else:
            static_nodes = torch.tensor([[]])

        q = self._extract_seq_data(q)
        p = self._extract_seq_data(p)
        q_next = self._extract_seq_data(q_next)
        p_next = self._extract_seq_data(p_next)
        u = self._extract_seq_data(u)
        dq_dt=self._extract_seq_data(dq_dt)
        dp_dt=self._extract_seq_data(dp_dt)
        q_noiseless=self._extract_seq_data(q_noiseless)
        p_noiseless=self._extract_seq_data(p_noiseless)
        t = self._extract_seq_data(t)

        # Package and return
        return Trajectory(name=name, trajectory_meta=meta,
                          q=q if self.seq_len > 1 else q.squeeze(0),
                          p=p if self.seq_len > 1 else p.squeeze(0),
                          q_next = q_next if self.seq_len > 1 else q_next.squeeze(0),
                          p_next = p_next if self.seq_len > 1 else p_next.squeeze(0),
                          u = u if self.seq_len > 1 else u.squeeze(0),
                          dq_dt=dq_dt if self.seq_len > 1 else dq_dt.squeeze(0),
                          dp_dt=dp_dt if self.seq_len > 1 else dp_dt.squeeze(0),
                          dt=dt,
                          t=t if self.seq_len > 1 else t.squeeze(0),
                          q_noiseless=q_noiseless if self.seq_len > 1 else q_noiseless.squeeze(0),
                          p_noiseless=p_noiseless if self.seq_len > 1 else p_noiseless.squeeze(0),
                          masses=masses,
                          edge_index=edge_index,
                          vertices=vertices,
                          fixed_mask_q=self.linearize(fixed_mask_q),
                          fixed_mask_p=self.linearize(fixed_mask_p),
                          extra_fixed_mask=self.linearize(extra_fixed_mask),
                          static_nodes=self.linearize(static_nodes),
                          sys_id=sys_id
                          )

    def __len__(self):
        return len(self._trajectory_meta)


Snapshot = namedtuple("Snapshot", ["name", "p", "q", "dp_dt", "dq_dt",
                                   "t", "trajectory_meta",
                                   "p_noiseless", "q_noiseless",
                                   "masses", "edge_index", "vertices",
                                   "fixed_mask_p", "fixed_mask_q",
                                   "extra_fixed_mask", "static_nodes",
                                   ])

class SnapshotDataset(data.Dataset):

    def __init__(self, traj_dataset):
        super().__init__()
        self._traj_dataset = traj_dataset

        self.system = self._traj_dataset.system
        self.system_metadata = self._traj_dataset.system_metadata

        name = []
        p = []
        q = []
        dp_dt = []
        dq_dt = []
        t = []
        traj_meta = []
        p_noiseless = []
        q_noiseless = []
        masses = []
        edge_indices = []
        vertices = []
        fixed_mask_p = []
        fixed_mask_q = []
        extra_fixed_mask = []
        static_nodes = []

        for traj_i in range(len(self._traj_dataset)):
            traj = self._traj_dataset[traj_i]
            # Stack the components
            traj_num_steps = traj.p.shape[0]
            name.extend([traj.name] * traj_num_steps)
            p.append(traj.p)
            q.append(traj.q)
            dp_dt.append(traj.dp_dt)
            dq_dt.append(traj.dq_dt)
            t.append(traj.t)
            traj_meta.extend([traj.trajectory_meta] * traj_num_steps)
            p_noiseless.append(traj.p_noiseless)
            q_noiseless.append(traj.q_noiseless)
            masses.extend([traj.masses] * traj_num_steps)
            edge_indices.extend([traj.edge_index] * traj_num_steps)
            vertices.extend([traj.vertices] * traj_num_steps)
            # Remove fake time dimension added above
            fixed_mask_p.extend([traj.fixed_mask_p[0]] * traj_num_steps)
            fixed_mask_q.extend([traj.fixed_mask_q[0]] * traj_num_steps)
            extra_fixed_mask.extend([traj.extra_fixed_mask[0]] * traj_num_steps)
            static_nodes.extend([traj.static_nodes[0]] * traj_num_steps)

        # Load each trajectory and join the components
        self._name = name
        self._p = np.concatenate(p)
        self._q = np.concatenate(q)
        self._dp_dt = np.concatenate(dp_dt)
        self._dq_dt = np.concatenate(dq_dt)
        self._t = np.concatenate(t)
        self._traj_meta = traj_meta
        self._p_noiseless = np.concatenate(p_noiseless)
        self._q_noiseless = np.concatenate(q_noiseless)
        self._masses = masses
        self._edge_indices = edge_indices
        self._vertices = vertices
        self._fixed_mask_p = fixed_mask_p
        self._fixed_mask_q = fixed_mask_q
        self._extra_fixed_mask = extra_fixed_mask
        self._static_nodes = static_nodes

    def __getitem__(self, idx):
        return Snapshot(name=self._name[idx],
                        trajectory_meta=self._traj_meta[idx],
                        p=self._p[idx], q=self._q[idx],
                        dp_dt=self._dp_dt[idx], dq_dt=self._dq_dt[idx],
                        t=self._t[idx],
                        p_noiseless=self._p_noiseless[idx],
                        q_noiseless=self._q_noiseless[idx],
                        masses=self._masses[idx],
                        edge_index=self._edge_indices[idx],
                        vertices=self._vertices[idx],
                        fixed_mask_p=self._fixed_mask_p[idx],
                        fixed_mask_q=self._fixed_mask_q[idx],
                        extra_fixed_mask=self._extra_fixed_mask[idx],
                        static_nodes=self._static_nodes[idx],
                        )

    def __len__(self):
        return len(self._traj_meta)



StepSnapshot = namedtuple("StepSnapshot",
                          ["name", "p", "q", "dp_dt", "dq_dt",
                           "p_step", "q_step",
                           "t", "trajectory_meta",
                           "p_noiseless", "q_noiseless",
                           "masses", "edge_index", "vertices",
                           "fixed_mask_p", "fixed_mask_q",
                           "extra_fixed_mask", "static_nodes",
                           ])


class StepSnapshotDataset(data.Dataset):

    def __init__(self, traj_dataset, subsample=1, time_skew=1):
        self._traj_dataset = traj_dataset
        self.subsample = subsample
        self.time_skew = time_skew

        self.system = self._traj_dataset.system
        self.system_metadata = self._traj_dataset.system_metadata

        name = []
        p = []
        q = []
        dp_dt = []
        dq_dt = []
        t = []
        traj_meta = []
        p_noiseless = []
        q_noiseless = []
        masses = []
        edge_indices = []
        vertices = []
        fixed_mask_p = []
        fixed_mask_q = []
        extra_fixed_mask = []
        static_nodes = []

        for traj_i in range(len(self._traj_dataset)):
            traj = self._traj_dataset[traj_i]
            # Stack the components
            traj_num_steps = math.ceil((traj.p.shape[0] - time_skew) / subsample)
            name.extend([traj.name] * traj_num_steps)
            p.append(traj.p[:-self.time_skew:self.subsample])
            q.append(traj.q[:-self.time_skew:self.subsample])
            dp_dt.append(traj.p[self.time_skew::self.subsample])
            dq_dt.append(traj.q[self.time_skew::self.subsample])
            t.append(traj.t[:-self.time_skew:self.subsample])
            traj_meta.extend([traj.trajectory_meta] * traj_num_steps)
            p_noiseless.append(traj.p_noiseless[:-self.time_skew:self.subsample])
            q_noiseless.append(traj.q_noiseless[:-self.time_skew:self.subsample])
            masses.extend([traj.masses] * traj_num_steps)
            edge_indices.extend([traj.edge_index] * traj_num_steps)
            vertices.extend([traj.vertices] * traj_num_steps)
            # Remove fake time dimension added above
            fixed_mask_p.extend([traj.fixed_mask_p[0]] * traj_num_steps)
            fixed_mask_q.extend([traj.fixed_mask_q[0]] * traj_num_steps)
            extra_fixed_mask.extend([traj.extra_fixed_mask[0]] * traj_num_steps)
            static_nodes.extend([traj.static_nodes[0]] * traj_num_steps)
            # Check length computation
            assert p[-1].shape[0] == traj_num_steps

        # Load each trajectory and join the components
        self._name = name
        self._p = np.concatenate(p)
        self._q = np.concatenate(q)
        self._dp_dt = np.concatenate(dp_dt)
        self._dq_dt = np.concatenate(dq_dt)
        self._t = np.concatenate(t)
        self._traj_meta = traj_meta
        self._p_noiseless = np.concatenate(p_noiseless)
        self._q_noiseless = np.concatenate(q_noiseless)
        self._masses = masses
        self._edge_indices = edge_indices
        self._vertices = vertices
        self._fixed_mask_p = fixed_mask_p
        self._fixed_mask_q = fixed_mask_q
        self._extra_fixed_mask = extra_fixed_mask
        self._static_nodes = static_nodes

    def __getitem__(self, idx):
        p_step = self._dp_dt[idx]
        q_step = self._dq_dt[idx]
        return StepSnapshot(name=self._name[idx],
                        trajectory_meta=self._traj_meta[idx],
                        p=self._p[idx], q=self._q[idx],
                        dp_dt=p_step, dq_dt=q_step,
                        p_step=p_step, q_step=q_step,
                        t=self._t[idx],
                        p_noiseless=self._p_noiseless[idx],
                        q_noiseless=self._q_noiseless[idx],
                        masses=self._masses[idx],
                        edge_index=self._edge_indices[idx],
                        vertices=self._vertices[idx],
                        fixed_mask_p=self._fixed_mask_p[idx],
                        fixed_mask_q=self._fixed_mask_q[idx],
                        extra_fixed_mask=self._extra_fixed_mask[idx],
                        static_nodes=self._static_nodes[idx],
                        )

    def __len__(self):
        return len(self._traj_meta)
