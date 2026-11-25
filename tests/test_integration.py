"""Automated End-to-End Integration Test Suite for CoolBuild AI."""

import os
import sys
import unittest
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_clean_dataset,
    get_feature_and_target_columns,
    derive_building_shape_groups,
    CSV_PATH
)
from src.preprocessing import validate_dataset, FEATURE_CONSTRAINTS
from src.train import predict_loads, load_trained_model, MODELS_DIR, MODEL_CONFIGS
from src.recommendation import (
    optimize_building_design,
    generate_feasible_alternatives,
    get_dataset_geometry_archetypes
)
from src.community_validation import (
    save_community_validation,
    load_all_validations,
    get_active_changeable_parameters
)


class TestCoolBuildIntegration(unittest.TestCase):
    """Full integration test suite covering data, models, optimizer, and community validation."""

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  STARTING COOLBUILD AI INTEGRATION TEST SUITE")
        print("=======================================================")

    def test_01_dataset_integrity(self):
        """Verify dataset row count, column count, no missing values, and physical conservation."""
        self.assertTrue(os.path.exists(CSV_PATH), f"Dataset CSV not found at {CSV_PATH}")
        df = load_clean_dataset()
        self.assertEqual(len(df), 768, "Dataset must contain exactly 768 rows")
        self.assertEqual(len(df.columns), 10, "Dataset must contain exactly 10 columns")
        
        # Data quality report
        report = validate_dataset(df)
        self.assertTrue(report["is_clean"], f"Dataset quality check failed: {report}")
        self.assertEqual(report["missing_values_total"], 0, "Missing values detected")
        self.assertEqual(report["duplicate_rows"], 0, "Duplicate rows detected")
        
        # Physical law: Surface Area == 2 * Roof Area + Wall Area
        diff = (df["surface_area"] - (2 * df["roof_area"] + df["wall_area"])).abs()
        self.assertAlmostEqual(diff.max(), 0.0, places=4, msg="Physical envelope conservation violated in dataset")

    def test_02_shape_grouping(self):
        """Verify exactly 12 building shapes with exactly 64 configurations each."""
        df = load_clean_dataset()
        shapes = derive_building_shape_groups(df)
        counts = shapes.value_counts()
        self.assertEqual(len(counts), 12, "Must identify exactly 12 geometry shapes")
        for shape_id, count in counts.items():
            self.assertEqual(count, 64, f"Shape {shape_id} must have exactly 64 configurations, got {count}")

    def test_03_model_artifacts_and_predictions(self):
        """Verify all trained model artifacts exist and produce valid positive predictions."""
        features, targets = get_feature_and_target_columns()
        
        # Check model files
        for target in targets:
            for algo_key in MODEL_CONFIGS.keys():
                path = os.path.join(MODELS_DIR, f"{target}_{algo_key}.pkl")
                self.assertTrue(os.path.exists(path), f"Missing model artifact: {path}")
            # Primary model
            primary_path = os.path.join(MODELS_DIR, f"{target}_model.pkl")
            self.assertTrue(os.path.exists(primary_path), f"Missing primary model: {primary_path}")

        # Test inference
        sample_bldg = {
            "relative_compactness": 0.74,
            "surface_area": 686.0,
            "wall_area": 245.0,
            "roof_area": 220.5,
            "overall_height": 3.5,
            "orientation": 3,
            "glazing_area": 0.25,
            "glazing_area_distribution": 1
        }
        
        for algo in ["random_forest", "decision_tree", "linear_regression", "svr"]:
            preds = predict_loads(sample_bldg, algorithm=algo)
            self.assertIn("heating_load", preds)
            self.assertIn("cooling_load", preds)
            self.assertGreater(preds["heating_load"], 5.0, f"Unreasonable heating load: {preds['heating_load']}")
            self.assertLess(preds["heating_load"], 50.0, f"Unreasonable heating load: {preds['heating_load']}")
            self.assertGreater(preds["cooling_load"], 10.0, f"Unreasonable cooling load: {preds['cooling_load']}")
            self.assertLess(preds["cooling_load"], 50.0, f"Unreasonable cooling load: {preds['cooling_load']}")

    def test_04_error_handling_invalid_inputs(self):
        """Verify proper error handling on invalid or incomplete inputs."""
        incomplete_input = {"relative_compactness": 0.74}
        with self.assertRaises(ValueError):
            predict_loads(incomplete_input)

    def test_05_community_validation_persistence(self):
        """Verify saving and retrieving stakeholder validation records."""
        rec = save_community_validation(
            stakeholder_name="Integration Test Architect",
            stakeholder_role="Architect / Building Designer",
            location="Test Site Block A",
            changeable_parameters=["glazing_area", "glazing_area_distribution"],
            observations="Orientation must be locked due to access road."
        )
        self.assertIsNotNone(rec.get("id"))
        
        changeable, fixed, is_prov, latest = get_active_changeable_parameters()
        self.assertFalse(is_prov, "Active status should not be provisional after saving a record")
        self.assertEqual(latest["stakeholder_name"], "Integration Test Architect")
        self.assertIn("glazing_area", changeable)
        self.assertIn("orientation", fixed)

    def test_06_recommendation_engine_constrained_optimization(self):
        """
        Verify that:
        1. Only changeable parameters are altered.
        2. Fixed parameters remain strictly unmodified.
        3. Cooling load reduction is non-negative.
        4. Recommended building obeys physical envelope conservation.
        """
        initial_building = {
            "relative_compactness": 0.76,
            "surface_area": 661.5,
            "wall_area": 416.5,
            "roof_area": 122.5,
            "overall_height": 7.0,
            "orientation": 2,
            "glazing_area": 0.40,
            "glazing_area_distribution": 2
        }
        
        # Test 1: Only Glazing is changeable
        res_glazing = optimize_building_design(
            initial_building,
            changeable_params=["glazing_area", "glazing_area_distribution"]
        )
        
        # Check cooling load reduction
        self.assertGreaterEqual(res_glazing["cooling_reduction"], 0.0)
        self.assertGreater(res_glazing["cooling_reduction"], 5.0, "Expected significant cooling reduction from reducing 40% glazing")
        
        # Check that fixed parameters were NOT modified
        rec_cfg = res_glazing["recommended_config"]
        self.assertEqual(rec_cfg["relative_compactness"], initial_building["relative_compactness"])
        self.assertEqual(rec_cfg["surface_area"], initial_building["surface_area"])
        self.assertEqual(rec_cfg["wall_area"], initial_building["wall_area"])
        self.assertEqual(rec_cfg["roof_area"], initial_building["roof_area"])
        self.assertEqual(rec_cfg["overall_height"], initial_building["overall_height"])
        self.assertEqual(rec_cfg["orientation"], initial_building["orientation"])
        
        # Check physical law on recommendation
        rec_diff = abs(rec_cfg["surface_area"] - (2 * rec_cfg["roof_area"] + rec_cfg["wall_area"]))
        self.assertAlmostEqual(rec_diff, 0.0, places=4, msg="Recommended configuration violates physical conservation")

    def test_07_evaluation_artifacts_exist(self):
        """Verify cross-validation and feature importance evaluation artifacts."""
        eval_files = [
            "cv_comparison_cooling_load.csv",
            "cv_comparison_heating_load.csv",
            "cv_comparison_all.json",
            "model_evaluation_report.md",
            "feature_importance_summary.csv",
            "feature_importance_report.md",
            "descriptive_statistics.csv"
        ]
        eval_dir = os.path.join(PROJECT_ROOT, "results", "evaluation")
        for f in eval_files:
            p = os.path.join(eval_dir, f)
            self.assertTrue(os.path.exists(p), f"Missing evaluation artifact: {p}")

    def test_08_plots_exist(self):
        """Verify all generated plots exist in results/plots/."""
        plot_files = [
            "01_feature_distributions.png",
            "02_target_distributions.png",
            "03_correlation_heatmap.png",
            "04_feature_vs_cooling_load.png",
            "05_feature_vs_heating_load.png",
            "06_heating_vs_cooling_coupling.png",
            "07_shape_groups_energy.png",
            "08_orientation_impact_analysis.png",
            "09_cv_cooling_load_r2_comparison.png",
            "10_cv_cooling_load_mae_comparison.png",
            "11_cv_fold_variance_boxplots.png",
            "12_rf_feature_importance_comparison.png",
            "13_permutation_importance_cooling.png",
            "14_permutation_importance_heating.png",
            "15_heating_vs_cooling_importance_scatter.png",
            "16_sample_building_optimization_comparison.png"
        ]
        plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")
        for f in plot_files:
            p = os.path.join(plots_dir, f)
            self.assertTrue(os.path.exists(p), f"Missing plot: {p}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
