"""
Prompts for the SeedDataGen pipeline.

All prompt templates are verbatim from project_plan.md.
Placeholders: {sample_text}, {conversation_history}, {conversation}.
"""

# -----------------------------------------------------------------------
# STEP 1 — QA generation prompt (system-only; model outputs QA pairs)
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# STEP 4 — User turn prompt (generates the next simulated user message)
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# STEP 4 — Assistant turn prompt (generates the assistant reply)
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# STEP 6 — LLM judge prompt (scores a conversation)
# -----------------------------------------------------------------------
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
