"""Chief-complaint pattern matching.

Triage chief complaints are three to eight words of clipped, abbreviation-dense text
typed under time pressure: "SOB x2d", "CP rad to L arm", "AMS, fam reports slurred
speech". A general-purpose tokenizer reads this as noise, so the red-flag layer needs
its own literal-minded matcher.

This lexicon is deliberately a *bootstrap*, not a clinical ontology. It exists so the
rule layer works on day one and so its recall can be measured. Before any real
deployment it is replaced by a curated mapping to SNOMED CT / the Emergency Care
Chief Complaint List, reviewed by the department's clinical lead — and the recall of
this version against that one is the number that justifies the swap.
"""

from __future__ import annotations

import re
from enum import Enum


class ComplaintFlag(str, Enum):
    ARREST = "arrest"
    STROKE = "stroke"
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    INFECTION = "infection"
    ALTERED_MENTAL_STATUS = "altered_mental_status"
    HEAD_INJURY = "head_injury"
    MAJOR_TRAUMA = "major_trauma"
    BLEEDING = "bleeding"
    OB_EMERGENCY = "ob_emergency"
    SELF_HARM = "self_harm"
    ANAPHYLAXIS = "anaphylaxis"
    SEIZURE = "seizure"


# Patterns are matched case-insensitively against the raw complaint string.
# `\b` boundaries keep "CP" from firing on "CPAP" and "SI" from firing on "sinus".
_PATTERNS: dict[ComplaintFlag, list[str]] = {
    # The patient in extremis often arrives with NO vitals at all — the monitor reads
    # nothing, the cuff cannot cycle, nobody has stopped compressions to take a pulse.
    # Every vital-sign rule is blind here, so the complaint text is the only signal
    # left and it must be enough on its own.
    ComplaintFlag.ARREST: [
        # Not a bare \barrest\b — police bring patients in "under arrest", and a
        # false level-1 on every custody arrival would destroy trust in the layer.
        r"\b(cardiac|respiratory|resp|circulatory)\s*arrest\b",
        r"\bcpr\b", r"\bvsa\b", r"\brosc\b",
        r"\bno\s*pulse\b", r"\bpulseless\b", r"\bcode\s*blue\b", r"\basystol",
        r"\bv[\s-]*fib\b", r"\bventricular\s*fibrillation\b", r"\bv[\s-]*tach\b",
        r"\bagonal\b", r"\bnot\s*breathing\b", r"\bunrespons\w*\b.*\bnot\s*breathing\b",
        r"\bperi[\s-]*arrest\b", r"\bcompressions\b", r"\bdefib",
    ],
    ComplaintFlag.STROKE: [
        r"\bstroke\b", r"\bcva\b", r"\btia\b",
        r"\bfacial\s*droop", r"\bslurr?ed\s*speech", r"\bdysarthri", r"\baphasi",
        r"\bweak(ness)?\s*(on\s*)?(one|1|l|r|left|right)\s*side",
        r"\b(left|right|l|r)\s*sided?\s*weak", r"\bhemipares", r"\bhemiplegi",
        r"\bnumbness\s*(one|1|l|r|left|right)\s*side", r"\bvision\s*loss\b",
        r"\bworst\s*headache\b", r"\bthunderclap\b",
    ],
    ComplaintFlag.CARDIAC: [
        r"\bcp\b", r"\bchest\s*(pain|pressure|tightness|discomfort|heaviness)",
        r"\bangina\b", r"\bmi\b", r"\bstemi\b", r"\bheart\s*attack\b",
        r"\bpalpitation", r"\bsubsternal\b", r"\bcardiac\b",
        r"\bpain\s*rad(iating)?\s*to\s*(l|left|r|right)?\s*(arm|jaw|shoulder)",
    ],
    ComplaintFlag.RESPIRATORY: [
        r"\bsob\b", r"\bshort(ness)?\s*of\s*breath", r"\bdyspn", r"\bresp\w*\s*distress",
        r"\bcan'?t\s*breath", r"\bdifficulty\s*breath", r"\bwheez", r"\bstridor\b",
        r"\bchoking\b", r"\bairway\b", r"\bapne", r"\bcyanos", r"\basthma\s*attack",
    ],
    ComplaintFlag.INFECTION: [
        r"\bfever\b", r"\bfebrile\b", r"\bsepsis\b", r"\bseptic\b", r"\binfect",
        r"\bcellulit", r"\bpneumonia\b", r"\bpna\b", r"\buti\b", r"\bpyelo",
        r"\bmeningit", r"\babscess\b", r"\bchills\b", r"\brigors\b", r"\bflu\b",
        r"\bcough\s*(and|&|\+)\s*fever",
    ],
    ComplaintFlag.ALTERED_MENTAL_STATUS: [
        r"\bams\b", r"\baltered\s*ment", r"\bconfus", r"\bunrespons", r"\bloc\b",
        r"\bloss\s*of\s*conscious", r"\bsyncop", r"\bpassed\s*out\b", r"\bfainted\b",
        r"\bdisoriented\b", r"\blethargic\b", r"\bobtund", r"\bnot\s*acting\s*right\b",
        r"\boverdose\b", r"\bod\b", r"\bintox",
    ],
    ComplaintFlag.HEAD_INJURY: [
        r"\bhead\s*(injury|trauma|strike|lac)", r"\bhit\s*head\b", r"\bstruck\s*head\b",
        r"\bhead\s*bleed\b", r"\btbi\b", r"\bconcussion\b", r"\bskull\b",
        r"\bfall\b.*\bhead\b", r"\bhead\b.*\bfall\b",
    ],
    ComplaintFlag.MAJOR_TRAUMA: [
        r"\bmvc\b", r"\bmva\b", r"\brta\b", r"\bmotor\s*vehicle\b", r"\bgsw\b",
        r"\bgun\s*shot\b", r"\bgunshot\b", r"\bstab", r"\bpenetrating\b",
        r"\bejected\b", r"\brollover\b", r"\bpedestrian\s*struck\b", r"\bfall\s*from\b",
        r"\bcrush\b", r"\bamputat", r"\bimpal", r"\bassault", r"\bburn\b",
    ],
    ComplaintFlag.BLEEDING: [
        r"\bbleed", r"\bhemorrhag", r"\bhaemorrhag", r"\bhematemesis\b",
        r"\bvomiting\s*blood\b", r"\bcoffee\s*ground", r"\bmelena\b", r"\bhematochezia\b",
        r"\bbrbpr\b", r"\bblood\s*in\s*(stool|vomit|urine)\b", r"\bepistaxis\b",
        r"\bhemoptysis\b", r"\buncontrolled\s*bleeding\b",
    ],
    ComplaintFlag.OB_EMERGENCY: [
        r"\bpregnan", r"\bpreg\b", r"\bg\d+p\d+\b", r"\bvaginal\s*bleed",
        r"\bcontractions\b", r"\blabor\b", r"\bmiscarriage\b", r"\bectopic\b",
        r"\bpreeclamp", r"\beclamp", r"\bwater\s*broke\b", r"\bdecreased\s*fetal\s*mov",
        r"\bpostpartum\b",
    ],
    ComplaintFlag.SELF_HARM: [
        r"\bsi\b", r"\bsuicid", r"\bself[\s-]*harm\b", r"\bcut\s*(my|him|her)self\b",
        r"\bwants?\s*to\s*die\b", r"\boverdose\s*intent", r"\bhomicidal\b", r"\bhi\b",
        r"\bpsych\s*eval\b", r"\bharm\s*(to\s*)?self\b",
    ],
    ComplaintFlag.ANAPHYLAXIS: [
        r"\banaphyla", r"\ballergic\s*react", r"\bhives\b.*\b(sob|breath|throat)",
        r"\bthroat\s*(closing|swelling|tight)", r"\btongue\s*swell", r"\bangioedema\b",
        r"\bbee\s*sting\b.*\bbreath", r"\bepi[\s-]*pen\b",
    ],
    ComplaintFlag.SEIZURE: [
        r"\bseizure", r"\bsz\b", r"\bconvuls", r"\bstatus\s*epilep", r"\bpostictal\b",
        r"\bfit\b", r"\bshaking\s*episode\b",
    ],
}

_COMPILED: dict[ComplaintFlag, re.Pattern[str]] = {
    flag: re.compile("|".join(patterns), re.IGNORECASE)
    for flag, patterns in _PATTERNS.items()
}


def flags_for(complaint: str) -> set[ComplaintFlag]:
    """Every clinical category a chief complaint touches. Empty set is a valid answer."""
    if not complaint or not complaint.strip():
        return set()
    return {flag for flag, pattern in _COMPILED.items() if pattern.search(complaint)}


def has_flag(complaint: str, flag: ComplaintFlag) -> bool:
    if not complaint or not complaint.strip():
        return False
    return bool(_COMPILED[flag].search(complaint))
