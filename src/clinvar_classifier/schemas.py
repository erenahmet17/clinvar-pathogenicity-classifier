"""GRCh38 genomic SNV and typed numeric annotation contracts."""

import math
import re
from dataclasses import asdict, dataclass

HGVS_RE = re.compile(
    r"^chr(?P<chrom>[1-9]|1[0-9]|2[0-2]|X|Y|M):g\.(?P<pos>[1-9][0-9]*)(?P<ref>[ACGT])>(?P<alt>[ACGT])$"
)


class ValidationError(ValueError):
    """Unsupported identity, ambiguous annotation, or invalid model input."""


def normalize_hgvs(value: str) -> str:
    value = str(value).strip()
    match = HGVS_RE.fullmatch(value)
    if not match or match["ref"] == match["alt"]:
        raise ValidationError(
            "Use a GRCh38 genomic SNV, e.g. chr11:g.126275389C>T; REF and ALT must differ."
        )
    return value


def finite_number(value, field: str, *, minimum=None, maximum=None):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        numbers = [finite_number(v, field, minimum=minimum, maximum=maximum) for v in value]
        present = {v for v in numbers if v is not None}
        if len(present) > 1:
            raise ValidationError(
                f"Ambiguous multiple values for {field}; explicit transcript policy required."
            )
        return next(iter(present), None)
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be numeric, not boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} is not numeric.") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite.")
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        raise ValidationError(f"{field} is outside supported bounds.")
    return number


@dataclass(frozen=True)
class Annotation:
    hgvs_id: str
    cadd_phred: float | None = None
    phylop_100way: float | None = None
    gnomad_af: float | None = None
    af_available: bool = False
    allele_count: float | None = None
    allele_number: float | None = None
    annotation_status: str = "ok"
    annotation_error: str | None = None

    def to_dict(self):
        return asdict(self)
