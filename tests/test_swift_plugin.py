import importlib.util
import pathlib
import sys
import types
import unittest

REWARDS_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tools" / "rewards.py"


def load_rewards_module(name: str):
    """Import rewards.py under a fresh name so registration runs again."""
    spec = importlib.util.spec_from_file_location(name, REWARDS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SwiftPluginRegistrationTest(unittest.TestCase):
    def _install_stub(self, layout: str):
        class ORM:
            pass

        orms = {}
        swift = types.ModuleType("swift")
        if layout == "v3":
            plugin = types.ModuleType("swift.plugin")
            plugin.ORM = ORM
            plugin.orms = orms
            swift.plugin = plugin
            modules = {"swift": swift, "swift.plugin": plugin}
        else:
            rewards = types.ModuleType("swift.rewards")
            rewards.ORM = ORM
            rewards.orms = orms
            swift.rewards = rewards
            modules = {"swift": swift, "swift.rewards": rewards}
        sys.modules.update(modules)
        return orms

    def _cleanup(self, layout: str):
        for name in ["swift", "swift.plugin", "swift.rewards"]:
            sys.modules.pop(name, None)

    def _check_layout(self, layout: str):
        orms = self._install_stub(layout)
        try:
            module = load_rewards_module(f"rewards_{layout}")
            self.assertIsNotNone(module.AGENTTHINK_REWARD_CLASS)
            self.assertIn("agentthink_reward", orms)

            instance = orms["agentthink_reward"](args=object())
            scores = instance(
                ["**Final Answer**: keep speed"], solution=["keep speed"]
            )
            self.assertEqual(len(scores), 1)
            self.assertGreaterEqual(scores[0], 0.0)
        finally:
            self._cleanup(layout)

    def test_ms_swift_3_layout(self):
        self._check_layout("v3")

    def test_ms_swift_4_layout(self):
        self._check_layout("v4")


if __name__ == "__main__":
    unittest.main()
