# Bundled robot resources

`curobo_franka/` contains the resources used by this project's Franka suction model:

- `assets/robot/franka_description/franka_suction_new.urdf`
- `assets/robot/franka_description/meshes/` (visual/collision meshes and materials)
- `configs/franka_suction_new.yml`
- `configs/spheres/franka_suction_new_mesh.yml`

Copied from the workspace's `curobo/playground/robots/curobo_franka` resources.
These are regular local copies, not links. Original resource notices are retained.
The YAML uses local URDF kinematics; unused external USD paths were removed.
Robot geometry, collision sphere values and motion settings are unchanged.
