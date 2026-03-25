"""pytest configuration — adds project root to sys.path."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
