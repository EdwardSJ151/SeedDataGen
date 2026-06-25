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
5. Não use referências meta ("o texto fornecido", "o trecho", "o chunk" ou números de chunk); escreva como conhecimento próprio e, ao citar a fonte, refira-se ao documento pelo nome.
6. Responda em português.

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
6. Não use referências meta ("o texto fornecido", "o trecho", "o chunk" ou números de chunk); ao citar a fonte, refira-se ao documento pelo nome.
7. Use português.

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
7. Não use referências meta ("o texto fornecido", "o trecho", "o chunk" ou números de chunk); ao citar a fonte, refira-se ao documento pelo nome.
8. Use português.

Texto:
{sample_text}

Tópicos já abordados em conversas anteriores sobre este texto (evite repeti-los):
{previous_questions}

Perguntas já feitas nesta conversa:
{conversation_history}

Próxima mensagem do usuário:"""


# STEP 4 — Assistant turn prompt (generates the assistant reply)
ASSISTANT_TURN_PROMPT = """\
Você é um assistente especializado e preciso.

Responda à última mensagem do usuário usando o conteúdo abaixo se referenciando a ele como se fosse seu próprio \
conhecimento.

Regras:
1. Use de forma prioritária as informações do conteúdo abaixo.
2. Não invente fatos.
3. Nunca use referências meta como "com base no texto fornecido", "no trecho \
fornecido", "nos trechos acima", "no contexto fornecido", "no chunk" ou números \
de chunk. Quando precisar citar a fonte, refira-se ao documento pelo nome.
4. Seja claro e direto.
5. Responda em português.

{sample_text}

Histórico:
{conversation_history}

Resposta:"""


# ---------------------------------------------------------------------------
# conv_expand_var — system/user split prompts.  Both turn-generators use the
# same structure: an immutable system persona + ONE user message carrying the
# chunk.  The chunk is rendered by utils.format_sample_text_for_prompt as
# <documento> blocks (no leaked "Chunk N" numbering).
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
    "dog_instruct": CONV_USER_TURN_USER_MSG,
    "qa_local_multihop": CONV_USER_TURN_MULTIHOP_USER_MSG,
    "qa_similarity_multihop": CONV_USER_TURN_MULTIHOP_USER_MSG,
}

CONV_EXPAND_USER_TURN_DIVERSITY_USER_MSG_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": CONV_USER_TURN_DIVERSITY_USER_MSG,
    "dog_instruct": CONV_USER_TURN_DIVERSITY_USER_MSG,
    "qa_local_multihop": CONV_USER_TURN_MULTIHOP_DIVERSITY_USER_MSG,
    "qa_similarity_multihop": CONV_USER_TURN_MULTIHOP_DIVERSITY_USER_MSG,
}
