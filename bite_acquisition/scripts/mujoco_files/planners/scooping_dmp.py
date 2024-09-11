import numpy as np
import matplotlib.pyplot as plt
import random
import pydmps
import pydmps.dmp_discrete
import quaternion
import os
try:
    from utils.transform_utils import mat2euler, quat2mat
except:
    from feeding_mujoco.utils.transform_utils import mat2euler, quat2mat

class ScoopingDMPPlanner:
    def __init__(self, dt=0.01, spoon_length=0.14):
        self._spoon_length = spoon_length

        demo_file = os.path.join("data", "scooping_bowl_left_new_tf.csv")   # Note this is already in wxyz
        print("Loading demo trajectory from file: ", demo_file)
        # check if file exists
        if not os.path.exists(demo_file):
            demo_file = os.path.join("../src/feeding_mujoco/data", "scooping_bowl_left_new_tf.csv")   # Note this is already in wxyz


        self._demo_traj = self.load_demo_traj(demo_file, change_to_wxyz=False)

        # Convert demo traj (eef traj) to spoon traj
        self._spoon_offset = np.array([0, 0, self._spoon_length])
        self._demo_spoon_traj = self.get_spoon_traj_from_eef_traj(self._demo_traj, self._spoon_offset)

        # Initialize dmp
        self._dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=7, n_bfs=100, dt=dt, ay=np.ones(7)*10.0)

        self._dmp.imitate_path(y_des=self._demo_spoon_traj.T, plot=False)
        
        # This initializes the original traj and bowl pos
        self.generate_orignal_dmp_traj()

        self._current_food_pose = np.array([0.5093849131582481, 
                                            -0.28866651753570133, 
                                            0.20004996849552092])

        self._ext_force = np.zeros(7)

        # Histories
        self._goal_hist = {}
        self._weight_scale_hist = {}
        self._traj = []
        self._stepped_traj = []

        self.reset()


    def load_demo_traj(self, file_path, change_to_wxyz=False):
        """
        Loads the demo trajectory from file

        Returns:
            ndarray: Numpy array of the demo trajectory in [x, y, z, w, x, y, z]
        """
        # Load the recorded traj from file
        demo_traj = np.loadtxt(file_path, delimiter=',')

        if change_to_wxyz:
            # Change orientation from {x, y, z, w} to {w, x, y, z}
            demo_traj[:, 3:7] = demo_traj[:, [6, 3, 4, 5]]

        return demo_traj
    
    def generate_orignal_dmp_traj(self):
        """
        Generates the original DMP trajectory
        """
        self._orig_y, self._orig_dy, self._orig_ddy = self._dmp.rollout()

        self._orig_bowl_pos = np.array([0.5093849131582481, 
                                        -0.28866651753570133, 
                                        0.20004996849552092])
        
        # Offset original bowl pos so that scooping pose is near the lowest mid of the trajectory
        self._orig_bowl_pos[0] += 0.04
        self._orig_bowl_pos[1] += 0.02
        self._orig_bowl_pos[2] -= 0.03

        self._orig_goal = self._dmp.goal
        self._orig_y0 = self._dmp.y0
        self._orig_w = self._dmp.w

        self._ref_goal = self._dmp.goal
        self._ref_y0 = self._dmp.y0
        self._ref_traj = self._orig_y
        self._ref_w = self._orig_w
        self.current_y = self._dmp.y0


    def update_food_pose(self, food_pose):

        # if current food pose is not the same as the new food pose
        if np.all(food_pose == self._current_food_pose[:3]):
            # print("Food pose is the same. No need to update path")
            return
        
        else:
            # Reset to imitate the original recorded traj
            self._dmp.imitate_path(y_des=self._demo_spoon_traj.T, plot=False)

            self._current_food_pose = food_pose

            # Offset to start and goal is scaled based on weight change and length change
            # First obtain the original vector from start to goal and start to bowl
            self._orig_start_to_goal = self._orig_goal[:3] - self._orig_y0[:3]
            self._orig_start_offset = self._orig_y0[:3] - self._orig_bowl_pos

            # Given a known offset from the scooping point, we calculate the new start
            scooping_length_scale = 0.7
            weight_change = 1.0
            start_offset = self._orig_start_offset * scooping_length_scale * weight_change
            self._dmp.y0[:3] = food_pose[:3] + start_offset

            # Scale the vector from start to goal to calculate the new goal
            scaled_start_to_goal = self._orig_start_to_goal * scooping_length_scale * weight_change
            self._dmp.goal[:3] = food_pose + start_offset + scaled_start_to_goal

            # need to create a new DMP with the new food pose 
            new_spoon_traj_to_learn = self.rollout(eef_traj=False)[0]       # output spoon traj
            self._dmp.imitate_path(y_des=new_spoon_traj_to_learn.T, plot=False)

            # Get the new trajectory
            y_ref, _, _ = self._dmp.rollout()

            # Store the desired start and goal for a given food pose
            # Needed to calculate an approximate target final pose
            self._ref_y0 = self._dmp.y0.copy()
            self._ref_goal = self._dmp.goal.copy()
            self._ref_traj = y_ref.copy()
            self._ref_w = self._dmp.w.copy()
            self._ref_traj_eef = self.get_eef_traj_from_spoon_traj(self._ref_traj, self._spoon_offset)
            
            # print(">> Updated dmp.y0 and dmp.goal [in spoon traj] based on food pose", food_pose)

            self._dmp.reset_state()
            return

    def update_dmp_params(self, params_dict):
        """
        Updates the DMP parameters

        Args:
            params_dict (dict): Dictionary containing the DMP parameters
                {
                    "delta_goal": ndarray of shape (3,) containing the change in goal in [x, y, z]
                    "delta_weight": float, scaling factor for the weight
                }
        """
        if 'delta_goal' in params_dict:
            # Scale goal by 0.01
            self._dmp.goal[:3] = self._ref_goal[:3] + (params_dict['delta_goal'][:3] * 0.01)
            self._goal_hist[self._step_idx] = self._dmp.goal

        elif 'delta_weight' in params_dict:
            # Convert range from -1.0 to 1.0 to 0.5 to 1.5
            # Scale action from -1 to 1 to 0 to 1
            scaled_action = (params_dict['delta_weight'] + 1.0) / 2.0
            # Offset scaled action to range of 0.5 to 1.5
            converted_action = scaled_action + 0.5
            self._dmp.w = self._ref_w * converted_action
            self._weight_scale_hist[self._step_idx] = converted_action
        
        elif 'fz' in params_dict:
            # Exert a force in the z direction
            scaled_force = params_dict['fz']
            self._ext_force = np.array([0, 0, scaled_force[0], 0, 0, 0, 0])

        elif 'fxyz' in params_dict:
            # Exert a force in the x, y, z direction
            scaled_force = params_dict['fxyz']
            self._ext_force = np.array([scaled_force[0], scaled_force[1], scaled_force[2], 0, 0, 0, 0])
        
        elif 'fz_discrete' in params_dict:
            # Exert a force in the z direction
            action_to_force_map = {
                0: 0.0,
                1: 1.0,
                2: 2.0,
                3: -1.0,
                4: -2.0
            }

            z_force = action_to_force_map[params_dict['fz_discrete']]
            self._ext_force = np.array([0, 0, z_force, 0, 0, 0, 0])
        
        elif 'noise' in params_dict:
            scaled_force = params_dict['noise']
            self._ext_force = np.array([scaled_force[0], scaled_force[1], scaled_force[2], 0, 0, 0, 0])

    def reset(self):
        """
        Resets the DMP planner
        """
        self._dmp.reset_state()

        self._goal_hist.clear()
        self._weight_scale_hist.clear()
        self._step_idx = 0
        self._stepped_traj = self._traj.copy()
        self.current_y = self._dmp.y0
        self._traj = []

        self.update_food_pose(self._current_food_pose)
    
    def step(self):
        """
        Steps the DMP planner and returns the next state

        Returns:
            y: Numpy array of the next eef_pose in [x, y, z, w, x, y, z]
            dy: Numpy array of the next eef_velocity in [x, y, z, w, x, y, z]
            ddy: Numpy array of the next eef_acceleration in [x, y, z, w, x, y, z]

        """
        self._step_idx += 1

        y, dy, ddy = self._dmp.step(
            external_force=self._ext_force
        )

        # Normalize the quaternions
        y[3:] = y[3:] / np.linalg.norm(y[3:])

        # Convert from spoon traj to eef traj
        y = self.get_eef_pose_from_spoon_pose(y, self._spoon_offset)

        # Check if quat is normalized
        assert np.isclose(np.linalg.norm(y[3:]), 1.0), "Quaternion is not normalized"
        
        self._traj.append(y.copy())

        self._ext_force = np.zeros(7)

        self._current_y = y
        
        return y, dy, ddy
    
    def get_stepped_traj(self):
        """
        Returns the stepped trajectory

        Returns:
            ndarray: Numpy array of the stepped trajectory in [x, y, z, w, x, y, z]
        """
        # if self._traj is empty, return self._stepped_traj
        if len(self._traj) == 0:
            return np.array(self._stepped_traj)
        else:
            return np.array(self._traj)
    
    def rollout(self, eef_traj=False):
        """
        Rolls out the DMP planner and returns the trajectory

        Returns:
            ndarray: Numpy array of the eef trajectory in [x, y, z, w, x, y, z]
            ndarray: Numpy array of the eef velocity in [x, y, z, w, x, y, z]
            ndarray: Numpy array of the eef acceleration in [x, y, z, w, x, y, z]
        """
        y, dy, ddy = self._dmp.rollout()

        if eef_traj:
            # Convert from spoon traj to eef traj
            y = self.get_eef_traj_from_spoon_traj(y, self._spoon_offset)
            dy = self.get_eef_traj_from_spoon_traj(dy, self._spoon_offset)
            ddy = self.get_eef_traj_from_spoon_traj(ddy, self._spoon_offset)
        return y, dy, ddy
    
    def change_scoop_length(self, start, end, length_scale):
        """
        Changes the scoop length by changing the start and end points

        Args:
            start (ndarray): Start point of the trajectory
            end (ndarray): End point of the trajectory
            length_scale (float): Scaling factor for the length
        """
        # get the vector from start to end
        vec = end[:3] - start[:3]
        
        # get midpoint
        midpoint = start[:3] + vec/2

        # scale the vector
        vec *= length_scale

        # update the start and end point
        # end[:3] = end[:3] - vec/2
        # start[:3] = start[:3] + vec/2
        end[:3] = midpoint + vec/2
        start[:3] = midpoint - vec/2

        return start, end
    
    def get_spoon_traj_from_eef_traj(self, eef_traj, spoon_offset):
        """
        Adds the spoon offset to the eef trajectory to get the spoon trajectory

        Args:
            eef_traj (ndarray): Numpy array of the end effectory trajectory in [x, y, z, w, x, y, z]
            spoon_offset (ndarray): Offset to be applied to the eef pose in [x, y, z]

        Returns:
            ndarray: Numpy array of the spoon trajectory in [x, y, z, w, x, y, z]
        """
        # translate demo_traj by spoon length in x-axis of eef frame
        spoon_traj = []

        for eef_pose in eef_traj:
            # translate position by spoon length
            spoon_pos = eef_pose[: 3] + spoon_offset

            # get rotation matrix of eef orientation
            eef_quat = eef_pose[3 :]
            eef_quat = np.quaternion(eef_quat[0], eef_quat[1], eef_quat[2], eef_quat[3])
            eef_rot_mat = quaternion.as_rotation_matrix(eef_quat)

            # calculate the actual spoon offset in world frame
            spoon_offset_world = eef_rot_mat.dot(spoon_offset)
            spoon_pos = eef_pose[: 3] + spoon_offset_world

            # spoon quat should be the same as eef_quat
            spoon_quat = eef_quat

            # update spoon_traj
            spoon_traj.append(np.hstack((spoon_pos, spoon_quat.components)))
        
        spoon_traj = np.array(spoon_traj)

        return spoon_traj

    def get_eef_traj_from_spoon_traj(self, spoon_traj, spoon_offset):
        """
        Removes the spoon offset from the spoon trajectory to get the eef trajectory

        Args:
            spoon_traj (ndarray): Numpy array of the spoon trajectory in [x, y, z, w, x, y, z]
            spoon_offset (ndarray): Offset to be applied to the spoon in [x, y, z]

        Returns:
            ndarray: Numpy array of the eef trajectory in [x, y, z, w, x, y, z]
        """
        # translate demo_traj by spoon length in x-axis of eef frame
        eef_traj = []

        for spoon_pose in spoon_traj:

            eef_pose = self.get_eef_pose_from_spoon_pose(spoon_pose, spoon_offset)
            eef_traj.append(eef_pose)
        
        
        eef_traj = np.array(eef_traj)

        return eef_traj
    
    def get_eef_pose_from_spoon_pose(self, spoon_pose, spoon_offset):
        """
        Removes the spoon offset from the spoon trajectory to get the eef trajectory

        Args:
            spoon_pose (ndarray): Numpy array of the spoon pose in [x, y, z, w, x, y, z]
            spoon_offset (ndarray): Offset to be applied to the spoon in [x, y, z]

        Returns:
            ndarray: Numpy array of the eef pose in [x, y, z, w, x, y, z]
        """
        # get rotation matrix of eef orientation
        spoon_pos = spoon_pose[: 3]
        spoon_quat = spoon_pose[3 :]
        spoon_quat = np.quaternion(spoon_quat[0], spoon_quat[1], spoon_quat[2], spoon_quat[3])
        spoon_rot_mat = quaternion.as_rotation_matrix(spoon_quat)

        # calculate eef position and orientation in world frame
        eef_offset_world = spoon_rot_mat.dot(spoon_offset)
        eef_pos = spoon_pos - eef_offset_world

        # spoon quat should be the same as eef_quat
        eef_quat = spoon_quat

        # update eef_traj
        eef_pose = np.hstack((eef_pos, eef_quat.components))
        
        return eef_pose


if __name__ == '__main__':
    sdp = ScoopingDMPPlanner()
    
    original_traj = sdp._orig_y
    print("Original traj shape:", original_traj.shape)
    
    sdp.reset()
    sdp.update_food_pose(np.array([0.4, 0.2, 0.15]))

    new_traj = sdp.rollout(eef_traj=True)[0]

    sdp.reset()

    traj2 = sdp.rollout(eef_traj=True)[0]

    sdp.reset()
    stepped_traj = []
    for i in range(sdp._dmp.timesteps):
        sdp.update_dmp_params({"delta_goal": np.array([0.0, 0.0, 0.0])})
        y, _, _ = sdp.step()
        stepped_traj.append(y.copy())
    
    stepped_traj = np.array(stepped_traj)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], label='original', marker='o')
    ax.plot(new_traj[:, 0], new_traj[:, 1], new_traj[:, 2], label='new', marker='o')
    ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], label='new2', marker='o')
    ax.plot(stepped_traj[:, 0], stepped_traj[:, 1], stepped_traj[:, 2], label='stepped', marker='o')

    spoon_traj_mod = sdp.get_spoon_traj_from_eef_traj(stepped_traj, sdp._spoon_offset)
    ax.plot(spoon_traj_mod[:, 0], spoon_traj_mod[:, 1], spoon_traj_mod[:, 2], label='spoon_traj_mod', marker='o')

    # Plot food pose
    ax.scatter(sdp._current_food_pose[0], sdp._current_food_pose[1], sdp._current_food_pose[2], c='r', marker='x', label='curr_food_pose')
    ax.scatter(sdp._orig_bowl_pos[0], sdp._orig_bowl_pos[1], sdp._orig_bowl_pos[2], c='g', marker='x', label='orig_bowl_pose')
    ax.scatter(spoon_traj_mod[0, 0], spoon_traj_mod[0, 1], spoon_traj_mod[0, 2], c='b', marker='x', label='initial')

    ax.legend()
    plt.show()

    # Plot orientation
    # Need to convert from quaternion to euler angles
    original_traj_euler = [mat2euler(quat2mat(x[3:])) for x in original_traj]
    new_traj_euler = [mat2euler(quat2mat(x[3:])) for x in new_traj]
    traj2_euler = [mat2euler(quat2mat(x[3:])) for x in traj2]
    stepped_traj_euler = [mat2euler(quat2mat(x[3:])) for x in stepped_traj]


    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot roll (R)
    axs[0].plot(original_traj[:, 3], label='original', marker='o')
    axs[0].plot(new_traj[:, 3], label='new', marker='o')
    axs[0].plot(traj2[:, 3], label='new2', marker='o')
    axs[0].plot(stepped_traj[:, 3], label='stepped', marker='o')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Roll (R)')
    axs[0].legend()

    # Plot pitch (P)
    axs[1].plot(original_traj[:, 4], label='original', marker='o')
    axs[1].plot(new_traj[:, 4], label='new', marker='o')
    axs[1].plot(traj2[:, 4], label='new2', marker='o')
    axs[1].plot(stepped_traj[:, 4], label='stepped', marker='o')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Pitch (P)')
    axs[1].legend()

    # Plot yaw (Y)
    axs[2].plot(original_traj[:, 5], label='original', marker='o')
    axs[2].plot(new_traj[:, 5], label='new', marker='o')
    axs[2].plot(traj2[:, 5], label='new2', marker='o')
    axs[2].plot(stepped_traj[:, 5], label='stepped', marker='o', alpha=0.5)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Yaw (Y)')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
