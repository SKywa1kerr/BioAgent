"""Data models for primer design."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class Primer:
    """A designed primer."""
    sequence: str
    tm: float  # melting temperature in °C
    gc_content: float  # GC percentage
    position: int  # start position in reference sequence (0-based)
    direction: Literal["forward", "reverse"]
    name: str = ""  # e.g., "F1", "R2"


@dataclass
class MutationTarget:
    """Target site for mutagenesis."""
    position: int  # codon position (0-based)
    original_codon: str
    target_codons: List[str]  # list of alternative codons for SSSM
    strategy: Literal["SSSM", "MSDM", "PAS"]


@dataclass
class PrimerResult:
    """Result of a primer design workflow."""
    primers: List[Primer]
    targets: List[MutationTarget]
    workflow: str  # e.g., "SSSM", "MSDM", "PAS"
    warnings: List[str] = field(default_factory=list)