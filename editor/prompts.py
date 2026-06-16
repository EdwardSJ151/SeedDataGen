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

Instrução de estilo para sua próxima pergunta:
{style_instruction}

Gere a próxima mensagem do usuário seguindo o estilo acima.

Regras:
1. Continue a conversa atual.
2. Faça exatamente uma pergunta ou pedido de esclarecimento, seguindo o estilo indicado.
3. Não repita perguntas anteriores desta conversa.
4. Explore um aspecto do texto que ainda não tenha sido abordado nas conversas anteriores listadas abaixo.
5. Mantenha-se no conteúdo do texto.
6. Seja natural e breve.
7. Use português.
8. Não mencione o nome do estilo na mensagem.

Texto:
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

Instrução de estilo para sua próxima pergunta:
{style_instruction}

Gere a próxima mensagem do usuário seguindo o estilo acima.

Regras:
1. Continue a conversa atual.
2. Faça exatamente uma pergunta ou pedido de esclarecimento, seguindo o estilo indicado.
3. Não repita perguntas anteriores.
4. Mantenha-se no conteúdo do texto.
5. Seja natural e breve.
6. Use português.
7. Não mencione o nome do estilo na mensagem.

Texto:
{sample_text}

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
    "qa_local_multihop": USER_TURN_VAR_PROMPT,
    "qa_similarity_multihop": USER_TURN_VAR_PROMPT,
}


# Maps GEN_TYPE → assistant-turn prompt template.  Falls back to
# ASSISTANT_TURN_PROMPT.  Templates accept sample_text + conversation_history.
CONV_EXPAND_ASSISTANT_TURN_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var": ASSISTANT_TURN_PROMPT,
    "qa_local_multihop": ASSISTANT_TURN_PROMPT,
    "qa_similarity_multihop": ASSISTANT_TURN_PROMPT,
}
