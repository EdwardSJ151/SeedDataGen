"""
Backwards-compatibility shim.

Prompts now live in per-role modules (``SeedDataGen.generator.prompts``,
``SeedDataGen.editor.prompts``, ``SeedDataGen.judge.prompts``).  This module
re-exports them so any legacy ``from SeedDataGen.prompts import X`` keeps
working.
"""

from SeedDataGen.generator.prompts import (  # noqa: F401
    DOG_INSTRUCT_QUESTION_SYSTEM_PROMPT,
    DOG_INSTRUCT_QUESTION_USER_PROMPT,
    DOG_INSTRUCT_REWRITE_SYSTEM_PROMPT,
    DOG_INSTRUCT_REWRITE_USER_PROMPT,
    QA_GENERATION_PROMPT,
    QA_GEN_VAR_SYSTEM_PROMPT,
    QA_GEN_VAR_STYLE_INSTRUCTIONS,
    QA_LOCAL_MULTIHOP_PROMPT,
    QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE,
    QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE,
)
from SeedDataGen.editor.prompts import (  # noqa: F401
    ANSWER_REWRITE_PROMPT,
    USER_TURN_PROMPT,
    USER_TURN_DIVERSITY_PROMPT,
    USER_TURN_VAR_PROMPT,
    USER_TURN_VAR_DIVERSITY_PROMPT,
    ASSISTANT_TURN_PROMPT,
    CONV_EXPAND_USER_TURN_BY_GEN_TYPE,
    CONV_EXPAND_ASSISTANT_TURN_BY_GEN_TYPE,
)
from SeedDataGen.judge.prompts import JUDGE_PROMPT  # noqa: F401
