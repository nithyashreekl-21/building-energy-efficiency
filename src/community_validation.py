"""Community stakeholder validation and feasibility persistence module for CoolBuild AI."""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    from src.data_loader import PROJECT_ROOT, PRETTY_FEATURE_NAMES
except ImportError:
    from data_loader import PROJECT_ROOT, PRETTY_FEATURE_NAMES

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VALIDATIONS_FILE = os.path.join(DATA_DIR, "community_validations.json")

STAKEHOLDER_ROLES = [
    "Mason / Local Contractor",
    "Architect / Building Designer",
    "Housing Office / Municipal Officer",
    "Civil Engineer / Structural Consultant",
    "Community Leader / Panchayat Representative",
    "Homeowner / Self-Builder"
]

DEFAULT_CHANGEABLE_PARAMS = ["glazing_area", "glazing_area_distribution"]


def load_all_validations() -> List[Dict[str, Any]]:
    """Load all recorded stakeholder validations from JSON store."""
    if not os.path.exists(VALIDATIONS_FILE):
        return []
    try:
        with open(VALIDATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_community_validation(
    stakeholder_name: str,
    stakeholder_role: str,
    location: str,
    changeable_parameters: List[str],
    observations: str,
    project_stage: str = "Pre-Construction / Planning"
) -> Dict[str, Any]:
    """
    Save a new local stakeholder validation record.
    Automatically assigns fixed parameters as the complement of changeable parameters.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    all_features = list(PRETTY_FEATURE_NAMES.keys())
    
    # Filter changeable to valid feature keys
    valid_changeable = [f for f in changeable_parameters if f in all_features]
    fixed_parameters = [f for f in all_features if f not in valid_changeable]
    
    record = {
        "id": f"VAL_{int(datetime.now().timestamp())}",
        "timestamp": datetime.now().isoformat(),
        "stakeholder_name": stakeholder_name.strip() if stakeholder_name else "Anonymous Stakeholder",
        "stakeholder_role": stakeholder_role,
        "location": location.strip() if location else "Unspecified Community Site",
        "project_stage": project_stage,
        "changeable_parameters": valid_changeable,
        "fixed_parameters": fixed_parameters,
        "observations": observations.strip() if observations else "No additional field notes provided."
    }
    
    validations = load_all_validations()
    validations.append(record)
    
    with open(VALIDATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(validations, f, indent=4)
        
    print(f"Community validation record saved: {record['id']} by {record['stakeholder_name']} ({record['stakeholder_role']})")
    return record


def get_latest_validation() -> Optional[Dict[str, Any]]:
    """Retrieve the most recent stakeholder validation record, or None if none entered."""
    validations = load_all_validations()
    if not validations:
        return None
    return validations[-1]


def get_active_changeable_parameters() -> Tuple[List[str], List[str], bool, Optional[Dict[str, Any]]]:
    """
    Get active changeable and fixed parameters.
    Returns:
    (changeable_params, fixed_params, is_provisional, latest_record)
    """
    latest = get_latest_validation()
    all_features = list(PRETTY_FEATURE_NAMES.keys())
    
    if latest is not None and "changeable_parameters" in latest and len(latest["changeable_parameters"]) > 0:
        changeable = latest["changeable_parameters"]
        fixed = [f for f in all_features if f not in changeable]
        is_provisional = False
        return changeable, fixed, is_provisional, latest
    else:
        # Default fallback is provisional
        changeable = list(DEFAULT_CHANGEABLE_PARAMS)
        fixed = [f for f in all_features if f not in changeable]
        is_provisional = True
        return changeable, fixed, is_provisional, None


if __name__ == "__main__":
    # Test saving a sample community validation
    sample_rec = save_community_validation(
        stakeholder_name="Ramesh Patel",
        stakeholder_role="Mason / Local Contractor",
        location="Sector 7 Community Housing Site",
        changeable_parameters=["glazing_area", "glazing_area_distribution"],
        observations="Plot boundary is tightly constrained so orientation and height are locked. Window sizes can be reduced or shaded easily."
    )
    changeable, fixed, prov, rec = get_active_changeable_parameters()
    print("\nActive Feasibility Configuration:")
    print(f"  Changeable: {changeable}")
    print(f"  Fixed: {fixed}")
    print(f"  Provisional Warning Required: {prov}")
