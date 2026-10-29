# Tool library (paper Appendix A)

The tool library is implemented in `scripts/tools/tool_libraries.py` and is
exposed to the model by `scripts/tools/function_schema.py`. The registry contains
exactly the 23 tools listed in Appendix A, in five groups. Coordinates are in the
ego-vehicle frame: `x` is lateral, `y` is longitudinal, units are metres.

## Visual information

| Tool | Parameters | Behaviour |
|---|---|---|
| `get_open_world_vocabulary_detection` | `object_names`, `image` | Open-vocabulary 2D boxes in the given camera/image |
| `get_3d_loc_in_cam` | `object_names`, `image` | Monocular depth + camera intrinsics → 3D location |
| `resize_image` | `input_path`, `output_path`, `target_size`, `interpolation?` | Resize with an optional interpolation method |
| `crop_image` | `input_path`, `output_path`, `box` | Crop `[left, upper, right, lower]` |

`image` accepts either a camera name (`CAM_FRONT`) or an image path; the agent
resolves camera names against the six camera paths of the current sample.

## Detection

| Tool | Parameters | Behaviour |
|---|---|---|
| `get_leading_object_detection` | – | Leading object within 10 m ahead and 6 m wide |
| `get_surrounding_object_detections` | – | Objects in the 20 m × 20 m box centred on the ego vehicle |
| `get_front_object_detections` | – | Objects in the 10 m × 20 m box ahead |
| `get_object_detections_in_range` | `x_start`, `x_end`, `y_start`, `y_end` | Objects inside an arbitrary rectangle |
| `get_all_object_detections` | – | All objects (fallback tool) |

## Prediction

| Tool | Parameters | Behaviour |
|---|---|---|
| `get_leading_object_future_trajectory` | – | Future trajectory of the leading object |
| `get_future_trajectories_specific_objects` | `object_ids` | Future trajectories for the given ids (`["o1", ...]`) |
| `get_future_trajectories_in_range` | `x_start`, `x_end`, `y_start`, `y_end` | Objects with **any waypoint** inside the rectangle |
| `get_future_waypoint_of_specific_objects_at_timestep` | `object_ids`, `timestep` | One waypoint per object at the requested step |
| `get_all_future_trajectories` | – | Trajectories of all objects (fallback tool) |

`get_future_trajectories_for_specific_objects` is kept as an alias for chains
generated with the v1 name.

## Occupancy

| Tool | Parameters | Behaviour |
|---|---|---|
| `get_occupancy_at_locations_for_timestep` | `locations`, `timestep` | Occupancy probability at `t ∈ [0, 4]`; `None` outside the scope |
| `check_occupancy_for_planned_trajectory` | `trajectory` | Collision check for a planned trajectory |

## Map

| Tool | Parameters | Behaviour |
|---|---|---|
| `get_drivable_at_locations` | `locations` | Drivability; `None` outside the map scope |
| `get_lane_category_at_locations` | `locations`, `return_score?` | Lane category (divider / ped crossing / boundary) |
| `get_distance_to_shoulder_at_locations` | `locations` | Distance to both road shoulders |
| `get_current_shoulder` | – | Shoulder distance at the ego position |
| `get_distance_to_lane_divider_at_locations` | `locations` | Distance to both lane dividers |
| `get_current_lane_divider` | – | Divider distance at the ego position |
| `get_nearest_pedestrian_crossing` | – | Location of the nearest pedestrian crossing |

## Schema conventions

- Every tool has a `*_info` dictionary with `name`, `description` and a JSON
  schema in `parameters`; `function_schema.py` turns these into the catalog shown
  to the model.
- Return values are `(prompt, data)` pairs: `prompt` is the natural-language
  observation inserted into the reasoning chain, `data` is the raw result used
  by downstream code.
- `None` is returned when the paper specifies "returns None" (no object, no
  crossing, out of map/occupancy scope).
- `tests/test_tool_library_contract.py` checks the registry against the Appendix
  A list and verifies that every registered tool has a callable method.
