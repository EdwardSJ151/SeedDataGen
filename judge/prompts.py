# Prompts used by JUDGE phases.

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
