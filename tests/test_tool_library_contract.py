import ast
import pathlib
import unittest

TOOL_LIBRARY = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tools" / "tool_libraries.py"

# Appendix A of the paper, grouped exactly as printed there.
PAPER_TOOLS = {
    "visual": [
        "get_open_world_vocabulary_detection",
        "get_3d_loc_in_cam",
        "resize_image",
        "crop_image",
    ],
    "detection": [
        "get_leading_object_detection",
        "get_surrounding_object_detections",
        "get_front_object_detections",
        "get_object_detections_in_range",
        "get_all_object_detections",
    ],
    "prediction": [
        "get_leading_object_future_trajectory",
        "get_future_trajectories_specific_objects",
        "get_future_trajectories_in_range",
        "get_future_waypoint_of_specific_objects_at_timestep",
        "get_all_future_trajectories",
    ],
    "occupancy": [
        "get_occupancy_at_locations_for_timestep",
        "check_occupancy_for_planned_trajectory",
    ],
    "map": [
        "get_drivable_at_locations",
        "get_lane_category_at_locations",
        "get_distance_to_shoulder_at_locations",
        "get_current_shoulder",
        "get_distance_to_lane_divider_at_locations",
        "get_current_lane_divider",
        "get_nearest_pedestrian_crossing",
    ],
}


def parse_library():
    tree = ast.parse(TOOL_LIBRARY.read_text(encoding="utf-8"))

    info_names = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        name = None
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == "name" and isinstance(value, ast.Constant):
                name = value.value
        if name:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    info_names[target.id] = name

    func_agent = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FuncAgent"
    )
    init = next(
        node for node in func_agent.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    registry = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr.endswith("_func_infos"):
                    group = target.attr[: -len("_func_infos")]
                    registry[group] = [
                        info_names.get(element.id, element.id)
                        for element in node.value.elts
                        if isinstance(element, ast.Name)
                    ]

    methods = {
        node.name for node in func_agent.body if isinstance(node, ast.FunctionDef)
    }
    return registry, info_names, methods


class ToolLibraryContractTest(unittest.TestCase):
    def setUp(self):
        self.registry, self.info_names, self.methods = parse_library()

    def test_registry_matches_appendix_a(self):
        self.assertEqual(set(self.registry), set(PAPER_TOOLS))
        for group, expected in PAPER_TOOLS.items():
            self.assertEqual(self.registry[group], expected, group)

    def test_registered_tools_are_callable(self):
        for names in self.registry.values():
            for name in names:
                self.assertIn(name, self.methods)

    def test_schema_names_are_unique(self):
        names = [name for names in self.registry.values() for name in names]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_tool_count(self):
        self.assertEqual(sum(len(names) for names in self.registry.values()), 23)


if __name__ == "__main__":
    unittest.main()
