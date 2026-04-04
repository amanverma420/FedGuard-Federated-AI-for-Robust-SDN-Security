"""
test_gui.py - Test GUI imports and basic functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all GUI imports work"""
    try:
        import streamlit as st
        print("✓ Streamlit imported")

        import config
        print("✓ Config imported")

        from data.data_loader import load_dataset
        print("✓ Data loader imported")

        from models.detector import IntrusionDetector
        print("✓ Detector imported")

        from federated.server import FederatedServer
        print("✓ Federated server imported")

        from adversarial.gan import AttackGAN
        print("✓ GAN imported")

        from dqn.agent import DQNAgent
        print("✓ DQN agent imported")

        print("\n🎉 All imports successful! GUI should work.")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()