# STEP 1 — QA generation prompt (system-only; model outputs QA pairs)
QA_GENERATION_PROMPT = """\
Tarefa:
Leia o texto, faça perguntas e responda a elas.

Siga estas instruções:
1. Faça perguntas diversas que cubram diferentes aspectos do texto.
2. Use diferentes tipos de pergunta:
  - sim/não
  - abertas (o que, como, por que, quando, onde)
  - comparação
  - compreensão
3. Foque em informações factuais e importantes.
4. Perguntas: claras e concisas.
5. Respostas: claras e completas para serem úteis como resposta de um assistente, sem ser prolixo.
6. Use texto simples e linguagem acessível.
7. Cada par deve estar em uma linha:
   "Pergunta: ... Resposta: ..."

Texto:
{sample_text}

Tarefa:
Gere até 5 pares pergunta-resposta."""


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

Histórico:
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

Histórico desta conversa:
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

Histórico desta conversa:
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

Histórico:
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


# QA_GEN_VAR — Style-constrained QA generation (one pair per style per document)
QA_GEN_VAR_SYSTEM_PROMPT = """\
Leia o texto abaixo e gere exatamente um par pergunta-resposta.

Instruções de estilo:
{style_instruction}

Regras gerais:
1. Foque em informações factuais e importantes do texto.
2. A pergunta deve ser clara.
3. A resposta deve ser clara, completa e útil como resposta de um assistente.
4. Use texto simples e linguagem acessível.
5. Responda em português.
6. Formate a resposta exatamente assim:
   Pergunta: ...
   Resposta: ...

Texto:
{sample_text}

Gere exatamente um par pergunta-resposta seguindo o estilo indicado."""


# Maps style name → Portuguese instruction string injected into QA_GEN_VAR_SYSTEM_PROMPT
QA_GEN_VAR_STYLE_INSTRUCTIONS: dict[str, str] = {
    "general": (
        "Faça uma pergunta geral que aborde o tema principal do texto. "
        "A pergunta deve ser respondível por qualquer pessoa que leu o texto inteiro."
    ),
    "specific": (
        "Faça uma pergunta focada em um detalhe específico, fato pontual ou dado concreto "
        "presente no texto. Evite perguntas amplas."
    ),
    "compositional": (
        "Faça uma pergunta que exija combinar ou relacionar duas ou mais informações "
        "distintas presentes no texto para ser respondida corretamente."
    ),
    "comparative": (
        "Faça uma pergunta que compare dois conceitos, entidades, períodos ou processos "
        "mencionados no texto, destacando semelhanças ou diferenças."
    ),
    "easy": (
        "Nível fácil: a pergunta deve ser direta e a resposta deve exigir pouca elaboração "
        "— em geral, localizar ou reformular uma ou duas ideias explícitas no texto, sem "
        "encadeamentos longos. Tudo deve estar claramente sustentado pelo texto."
    ),
    "medium": (
        "Nível médio: a pergunta pode exigir organizar informações espalhadas no texto ou "
        "um passo simples de inferência, mas sem argumentação longa. A resposta ainda deve "
        "ser construída só com o que o texto permite."
    ),
    "hard": (
        "Nível difícil: a pergunta deve exigir raciocínio mais denso — por exemplo, "
        "relacionar várias partes do texto, explicar um \"porquê\" ou um mecanismo implícito, "
        "ou sintetizar implicações sempre com base no que está escrito, sem extrapolar "
        "além do texto."
    ),
    "extra_hard": (
        "Nível extra difícil: a pergunta deve ser a mais exigente possível em termos de "
        "complexidade da resposta (múltiplos passos, sutilezas, comparações finas ou "
        "estruturação cuidadosa), desde que a resposta completa ainda possa ser fundamentada "
        "no texto. Não invente lacunas: se o texto não suporta um grau tão alto, aproxime-se "
        "do máximo que o texto ainda permita com rigor."
    ),
}


# STEP 6 — LLM judge prompt (scores a conversation)
JUDGE_PROMPT = """\
Você é um avaliador de qualidade de conversas para treinamento de IA.

Avalie a conversa com base no texto.

Critérios:
1. Fidelidade ao texto
2. Correção
3. Clareza
4. Coerência
5. Diversidade das interações

Dê notas de 1 a 5.

Formato:

Fidelidade: X
Correção: X
Clareza: X
Coerência: X
Diversidade: X

Texto:
{sample_text}

Conversa:
{conversation}"""
