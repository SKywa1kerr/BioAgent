"""Primer design skill implementations."""

from typing import List
from .primer_models import Primer, MutationTarget, PrimerResult
from .primer_skills import PrimerSkill, registry


def design_sssm_primers(sequence: str, targets: List[MutationTarget]) -> PrimerResult:
    """Design primers for Single-Site Saturation Mutagenesis."""
    raise NotImplementedError("Primer3 integration not yet implemented")


def design_msdm_primers(sequence: str, targets: List[MutationTarget]) -> PrimerResult:
    """Design primers for Multi-Site Directed Mutagenesis."""
    raise NotImplementedError("Primer3 integration not yet implemented")


def design_pas_primers(sequence: str, targets: List[MutationTarget]) -> PrimerResult:
    """Design primers for PCR-based Assembly."""
    raise NotImplementedError("Primer3 integration not yet implemented")


# Register skills with metadata
registry.register(PrimerSkill(
    name="design_sssm_primers",
    description="Design primers for Single-Site Saturation Mutagenesis",
    function=design_sssm_primers,
    input_schema={
        "sequence": {"type": "str", "description": "DNA sequence"},
        "targets": {"type": "List[MutationTarget]", "description": "Mutation targets"}
    },
    output_schema={"type": "PrimerResult"},
    tags=["SSSM", "single-site", "point-mutation"]
))

registry.register(PrimerSkill(
    name="design_msdm_primers",
    description="Design primers for Multi-Site Directed Mutagenesis",
    function=design_msdm_primers,
    input_schema={
        "sequence": {"type": "str", "description": "DNA sequence"},
        "targets": {"type": "List[MutationTarget]", "description": "Mutation targets"}
    },
    output_schema={"type": "PrimerResult"},
    tags=["MSDM", "multi-site", "directed-mutagenesis"]
))

registry.register(PrimerSkill(
    name="design_pas_primers",
    description="Design primers for PCR-based Assembly",
    function=design_pas_primers,
    input_schema={
        "sequence": {"type": "str", "description": "DNA sequence"},
        "targets": {"type": "List[MutationTarget]", "description": "Mutation targets"}
    },
    output_schema={"type": "PrimerResult"},
    tags=["PAS", "assembly", "pcr"]
))