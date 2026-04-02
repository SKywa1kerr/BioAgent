"""Primer skill registry for discoverable primer design functions."""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional


@dataclass
class PrimerSkill:
    """Metadata for a primer design skill."""
    name: str
    description: str
    function: Callable
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tags: List[str]  # e.g., ["SSSM", "single-site", "point-mutation"]


class PrimerSkillRegistry:
    """Registry of available primer design skills."""

    def __init__(self):
        self.skills: Dict[str, PrimerSkill] = {}

    def register(self, skill: PrimerSkill):
        """Register a new skill."""
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[PrimerSkill]:
        """Get a skill by name."""
        return self.skills.get(name)

    def find_skills_by_tags(self, tags: List[str]) -> List[PrimerSkill]:
        """Find skills matching all given tags."""
        if not tags:
            return list(self.skills.values())
        return [s for s in self.skills.values() if all(tag in s.tags for tag in tags)]

    def list_skills(self) -> List[PrimerSkill]:
        """List all registered skills."""
        return list(self.skills.values())


# Global registry instance
registry = PrimerSkillRegistry()