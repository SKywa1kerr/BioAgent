"""CLI entry point for primer design skills."""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

# Import skills to ensure they're registered
from . import primer_design
from .primer_skills import registry
from .primer_models import MutationTarget, PrimerResult


def skill_to_dict(skill: Any) -> dict:
    """Convert a skill to a JSON-serializable dictionary."""
    return {
        "name": skill.name,
        "description": skill.description,
        "input_schema": skill.input_schema,
        "tags": skill.tags
    }


def main():
    parser = argparse.ArgumentParser(description="Primer design CLI")
    parser.add_argument("--skill", help="Skill name to execute")
    parser.add_argument("--params", help="JSON parameters for the skill")
    parser.add_argument("--list-skills", action="store_true", help="List available skills")

    args = parser.parse_args()

    if args.list_skills:
        skills = registry.list_skills()
        result = [skill_to_dict(s) for s in skills]
        print(json.dumps(result, separators=(",", ":")))
        return

    if not args.skill or not args.params:
        print(json.dumps({"error": "Both --skill and --params are required"}))
        sys.exit(1)

    skill = registry.get_skill(args.skill)
    if not skill:
        print(json.dumps({"error": f"Skill not found: {args.skill}"}))
        sys.exit(1)

    try:
        params = json.loads(args.params)
        # Convert params dict to appropriate arguments
        # For now assume params match function signature
        result = skill.function(**params)

        # Convert result to JSON
        if hasattr(result, "__dict__"):
            result_dict = asdict(result)
        else:
            result_dict = result

        print(json.dumps(result_dict, separators=(",", ":")))

    except NotImplementedError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Skill execution failed: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()