# Prompts used by EDITOR phases (answer_rewrite, conv_expand, conv_expand_var).

# STEP 1B — Answer rewrite prompt (optional; enriches Phase 1 answers before filtering)
ANSWER_REWRITE_PROMPT = """\
Você tem uma pergunta, uma resposta curta e o texto original de onde ela foi extraída.

Sua tarefa é reescrever a resposta para torná-la mais completa e informativa, \
sem adicionar informações que não estejam no texto.

Regras:
1. Use apenas informações presentes no texto.
2. Não invente, não extrapole.
3. Mantenha o foco na pergunta — não desvie para outros tópicos.
4. A resposta deve ser mais rica que a original, mas ainda direta e objetiva.
5. Responda em português.

Texto:
{sample_text}

Pergunta:
{question}

Resposta original:
{answer}

Resposta reescrita:"""


# STEP 4 — User turn prompt (generates the next simulated user message)
USER_TURN_PROMPT = """\
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

Gere a próxima mensagem do usuário.

Regras:
1. Continue a conversa atual.
2. Faça uma pergunta, dúvida ou pedido de esclarecimento.
3. Não repita perguntas anteriores.
4. Mantenha-se no conteúdo do texto.
5. Seja natural e breve.
6. Use português.

Texto:
{sample_text}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# CONV_EXPAND (non-naive) — Diversity-aware user turn prompt
# Used when naive_gen=False.  previous_questions lists the seed questions already
# covered by earlier conversations from the same source document, so the model
# is steered away from repeating already-covered ground.
# previous_questions is omitted entirely when there are no prior conversations.
USER_TURN_DIVERSITY_PROMPT = """\
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

Gere a próxima mensagem do usuário.

Regras:
1. Continue a conversa atual.
2. Faça uma pergunta, dúvida ou pedido de esclarecimento.
3. Não repita perguntas anteriores desta conversa.
4. Explore um aspecto do texto que ainda não tenha sido abordado nas conversas anteriores listadas abaixo.
5. Mantenha-se no conteúdo do texto.
6. Seja natural e breve.
7. Use português.

Texto:
{sample_text}

Tópicos já abordados em conversas anteriores sobre este texto (evite repeti-los):
{previous_questions}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# CONV_EXPAND_VAR (diversity mode) — Style-aware + diversity-aware user turn prompt
# Used when naive_gen=False and there are prior seed questions from the same sample.
# Combines style constraint with cross-conversation diversity steering.
# The assistant never sees this prompt.
USER_TURN_VAR_DIVERSITY_PROMPT = """\
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

Regras:
1. Continue a conversa atual.
2. Faça exatamente uma pergunta ou pedido de esclarecimento, seguindo o estilo indicado.
3. Não repita perguntas anteriores desta conversa.
4. Explore um aspecto do texto que ainda não tenha sido abordado nas conversas anteriores listadas abaixo.
5. Mantenha-se no conteúdo do texto.
6. Seja natural e breve.
7. Use português.
8. Não mencione o nome do estilo na mensagem.

Instrução de estilo para sua próxima pergunta:
{style_instruction}

{doc_summary}Texto:
{sample_text}

Tópicos já abordados em conversas anteriores sobre este texto (evite repeti-los):
{previous_questions}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# CONV_EXPAND_VAR — Style-aware user turn prompt
# The style_instruction is injected here so the simulated user stays on-style.
# The assistant never sees this prompt or the style, it only sees the conversation history.
USER_TURN_VAR_PROMPT = """\
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

Regras:
1. Continue a conversa atual.
2. Faça exatamente uma pergunta ou pedido de esclarecimento, seguindo o estilo indicado.
3. Não repita perguntas anteriores.
4. Mantenha-se no conteúdo do texto.
5. Seja natural e breve.
6. Use português.
7. Não mencione o nome do estilo na mensagem.

Instrução de estilo para sua próxima pergunta:
{style_instruction}

{doc_summary}Texto:
{sample_text}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# CONV_EXPAND_VAR — Style-aware user turn prompt (multihop: multiple labelled chunks)
USER_TURN_VAR_MULTIHOP_PROMPT = """\
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

O texto abaixo contém vários trechos distintos, cada um marcado com [Chunk ...].
Quando possível, faça perguntas que usem ou relacionem informações de mais de um trecho.

Regras:
1. Continue a conversa atual.
2. Faça exatamente uma pergunta ou pedido de esclarecimento, seguindo o estilo indicado.
3. Não repita perguntas anteriores.
4. Mantenha-se no conteúdo dos trechos abaixo.
5. Seja natural e breve.
6. Use português.
7. Não mencione o nome do estilo na mensagem.

Instrução de estilo para sua próxima pergunta:
{style_instruction}

{doc_summary}Texto (múltiplos trechos):
{sample_text}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# CONV_EXPAND_VAR (diversity mode) — multihop variant
USER_TURN_VAR_MULTIHOP_DIVERSITY_PROMPT = """\
Você está simulando um usuário aprendendo sobre o conteúdo abaixo.

O texto abaixo contém vários trechos distintos, cada um marcado com [Chunk ...].
Dê prioridade em fazer perguntas que usem ou relacionem informações de mais de um trecho.

Regras:
1. Continue a conversa atual.
2. Faça exatamente uma pergunta ou pedido de esclarecimento, seguindo o estilo indicado.
3. Não repita perguntas anteriores desta conversa.
4. Explore um aspecto dos trechos que ainda não tenha sido abordado nas conversas anteriores listadas abaixo.
5. Mantenha-se no conteúdo dos trechos abaixo.
6. Seja natural e breve.
7. Use português.
8. Não mencione o nome do estilo na mensagem.

Instrução de estilo para sua próxima pergunta:
{style_instruction}

{doc_summary}Texto (múltiplos trechos):
{sample_text}

Tópicos já abordados em conversas anteriores sobre este texto (evite repeti-los):
{previous_questions}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# STEP 4 — Assistant turn prompt (generates the assistant reply)
ASSISTANT_TURN_PROMPT = """\
Você é um assistente de IA útil e preciso.

Responda à última mensagem do usuário com base no texto abaixo.

Regras:
1. Use de forma prioritária informações do texto.
2. Não invente fatos.
3. Seja claro e direto.
4. Responda em português.

Texto:
{sample_text}

Histórico:
{conversation_history}

Resposta:"""


# CONV_EXPAND — placeholder prompt routing by GEN_TYPE.
# Maps GEN_TYPE → user-turn prompt template.  Falls back to USER_TURN_VAR_PROMPT
# when a GEN_TYPE is not present here.  Templates accept the same fields as
# USER_TURN_VAR_PROMPT (style_instruction, sample_text, conversation_history).
CONV_EXPAND_USER_TURN_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": USER_TURN_VAR_PROMPT,
    "qa_local_multihop": USER_TURN_VAR_MULTIHOP_PROMPT,
    "qa_similarity_multihop": USER_TURN_VAR_MULTIHOP_PROMPT,
}

# Maps GEN_TYPE → diversity-mode user-turn prompt (naive_gen=False).
CONV_EXPAND_USER_TURN_DIVERSITY_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": USER_TURN_VAR_DIVERSITY_PROMPT,
    "qa_local_multihop": USER_TURN_VAR_MULTIHOP_DIVERSITY_PROMPT,
    "qa_similarity_multihop": USER_TURN_VAR_MULTIHOP_DIVERSITY_PROMPT,
}


# Maps GEN_TYPE → assistant-turn prompt template.  Falls back to
# ASSISTANT_TURN_PROMPT.  Templates accept sample_text + conversation_history.
CONV_EXPAND_ASSISTANT_TURN_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": ASSISTANT_TURN_PROMPT,
    "qa_local_multihop": ASSISTANT_TURN_PROMPT,
    "qa_similarity_multihop": ASSISTANT_TURN_PROMPT,
}


# ---------------------------------------------------------------------------
# conv_expand_var v2 — system/user split prompts (supersede the *_VAR_* strings
# above for the conv_expand_var phase).  Both turn-generators use the same
# structure: an immutable system persona + ONE user message carrying the chunk.
# The chunk is rendered by utils.format_sample_text_for_prompt as <documento>
# blocks (no leaked "Chunk N" numbering).
# ---------------------------------------------------------------------------

# Immutable persona + fixed behavioral rules for the simulated USER turn.
# Carries no run-dependent data.  The style instruction is NOT here (it varies
# per turn) — it goes in the user message.
CONV_USER_TURN_SYSTEM_PROMPT = """\
Você é um profissional curioso e prático que consulta um assistente \
especializado nas normas técnicas da CEMIG. Sua tarefa é escrever a próxima \
pergunta do profissional na conversa.

Regras invioláveis:
- Sua saída deve conter APENAS a pergunta — nenhuma saudação, explicação, \
resposta ou texto adicional.
- Escreva exatamente uma pergunta.
- Quando fizer referência à fonte, faça-o de forma natural pelo nome do \
documento (ex.: "Com base no documento ND-4.15, ...", "Considerando os \
procedimentos do documento X, ..."). NUNCA use referências meta como "com base \
no contexto fornecido", "no texto fornecido", "no trecho acima" ou números de chunk.
- Escreva em português."""

# Immutable persona + fixed anti-meta rule for the ASSISTANT turn.  Carries no
# run-dependent data; the refusal/bridging instruction lives in the user message.
CONV_ASSISTANT_TURN_SYSTEM_PROMPT = """\
Você é um assistente especializado nas normas técnicas da CEMIG.

Regras invioláveis:
- Nunca use referências meta como "com base no texto fornecido", "no trecho \
fornecido", "nos trechos acima", "no contexto fornecido" ou números de chunk. \
Quando precisar citar a fonte, refira-se ao documento pelo nome.
- Escreva em português."""


# USER-turn user message (mutable).  Fields: doc_summary, sample_text,
# style_instruction, conversation_history.
CONV_USER_TURN_USER_MSG = """\
{doc_summary}{sample_text}

Estilo da próxima pergunta:
{style_instruction}

Conversa até agora:
{conversation_history}

Escreva apenas a próxima pergunta do profissional:"""

# USER-turn user message, diversity mode.  Adds previous_questions.
CONV_USER_TURN_DIVERSITY_USER_MSG = """\
{doc_summary}{sample_text}

Estilo da próxima pergunta:
{style_instruction}

Tópicos já abordados em outras conversas sobre este documento (evite repeti-los):
{previous_questions}

Conversa até agora:
{conversation_history}

Escreva apenas a próxima pergunta do profissional, explorando um aspecto ainda não abordado:"""

# Multihop variants nudge cross-document questions when more than one document
# is present.  Same fields as the single-document variants.
CONV_USER_TURN_MULTIHOP_USER_MSG = """\
{doc_summary}{sample_text}

Os blocos acima podem conter mais de um documento. Quando fizer sentido, faça \
perguntas que relacionem informações de mais de um deles.

Estilo da próxima pergunta:
{style_instruction}

Conversa até agora:
{conversation_history}

Escreva apenas a próxima pergunta do profissional:"""

CONV_USER_TURN_MULTIHOP_DIVERSITY_USER_MSG = """\
{doc_summary}{sample_text}

Os blocos acima podem conter mais de um documento. Quando fizer sentido, faça \
perguntas que relacionem informações de mais de um deles.

Estilo da próxima pergunta:
{style_instruction}

Tópicos já abordados em outras conversas sobre este documento (evite repeti-los):
{previous_questions}

Conversa até agora:
{conversation_history}

Escreva apenas a próxima pergunta do profissional, explorando um aspecto ainda não abordado:"""


# ASSISTANT-turn user message (mutable).  Fields: sample_text,
# conversation_history, refusal_string.
CONV_ASSISTANT_TURN_USER_MSG = """\
{sample_text}

Conversa até agora:
{conversation_history}

Responda à última mensagem do usuário usando as informações dos documentos acima.
- Se os documentos tratarem o assunto apenas de forma tangencial, dê uma resposta \
parcial: diga, citando o documento pelo nome, o que ele cobre e o que não aborda.
- Se não houver nenhuma informação relevante, responda exatamente: {refusal_string}

Resposta:"""


# GEN_TYPE → user-message template (v2).  Multihop gen types get the cross-document
# nudge; everything else uses the single-document template.
CONV_EXPAND_USER_TURN_USER_MSG_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": CONV_USER_TURN_USER_MSG,
    "qa_local_multihop": CONV_USER_TURN_MULTIHOP_USER_MSG,
    "qa_similarity_multihop": CONV_USER_TURN_MULTIHOP_USER_MSG,
}

CONV_EXPAND_USER_TURN_DIVERSITY_USER_MSG_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": CONV_USER_TURN_DIVERSITY_USER_MSG,
    "qa_local_multihop": CONV_USER_TURN_MULTIHOP_DIVERSITY_USER_MSG,
    "qa_similarity_multihop": CONV_USER_TURN_MULTIHOP_DIVERSITY_USER_MSG,
}
